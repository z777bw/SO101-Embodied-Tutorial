#!/usr/bin/env python
"""LeRobotDataset v3.0 的最小读取实现（不依赖 lerobot / datasets 库）。

对应的磁盘布局（`meta/info.json` 里的 `codebase_version` 以 "v3" 开头）：

    <root>/
      meta/info.json                     # features / fps / 文件模板 / 总帧数
      meta/stats.json                    # 每个 feature 的 min/max/mean/std/count
      meta/tasks.parquet                 # task_index -> 任务文本（index 列即 task_index）
      meta/episodes/chunk-000/file-000.parquet
                                         # 每条 episode 一行：数据/视频分片位置、
                                         # 全局帧索引区间、视频时间偏移、每集统计量
      data/chunk-000/file-000.parquet    # 帧级表格数据（state / action / index …）
      videos/<video_key>/chunk-000/file-000.mp4

读取语义与 lerobot 0.6.2 `DatasetReader.get_item()` 对齐：
  * `idx` 是数据集内的行号（未做 episode 过滤时等于全局 `index`）；
  * `delta_timestamps` 会被换算成帧偏移，越界时**夹到 episode 边界**并给出
    `<key>_is_pad` 掩码（训练 ACT 的 action chunk 就是靠这个补齐的）；
  * 视频按"绝对时间戳 → episode 内 from_timestamp 偏移 → 最近帧"取帧，
    返回 (C, H, W) float32（[0, 1]），多帧时返回 (T, C, H, W)。

依赖：pyarrow（parquet）、numpy、torch、av（PyAV，可换成 torchcodec）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = ["FrameTimestampError", "LeRobotV3Dataset", "decode_video_frames_pyav"]

DEFAULT_TOLERANCE_S = 1e-4


class FrameTimestampError(ValueError):
    """请求的时间戳与视频里最近的一帧相差超过 tolerance_s。"""


class LeRobotV3Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = DEFAULT_TOLERANCE_S,
        video_backend: str = "pyav",
        return_uint8: bool = False,
    ):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found — is this a LeRobot dataset root?")
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))

        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v3"):
            raise ValueError(
                f"this reader implements the v3.0 layout, got codebase_version={version!r}. "
                "v2.1 datasets must be converted first "
                "(`python -m lerobot.scripts.convert_dataset_v21_to_v30`)."
            )

        self.fps = int(self.info["fps"])
        self.features: dict[str, dict] = self.info["features"]
        self.total_frames = int(self.info["total_frames"])
        self.total_episodes = int(self.info["total_episodes"])
        self.data_path_template = self.info["data_path"]
        self.video_path_template = self.info.get("video_path")

        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.return_uint8 = return_uint8

        self.video_keys = [k for k, ft in self.features.items() if ft["dtype"] == "video"]
        self.image_keys = [k for k, ft in self.features.items() if ft["dtype"] == "image"]
        self.camera_keys = self.video_keys + self.image_keys
        self.stats = self._load_stats()
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()

        # 每集在全局帧索引上的区间（用于 delta 越界判断）
        self._ep_from = {}
        self._ep_to = {}
        for ep in self.episodes:
            self._ep_from[int(ep["episode_index"])] = int(ep["dataset_from_index"])
            self._ep_to[int(ep["episode_index"])] = int(ep["dataset_to_index"])
        self._ep_order = sorted(self._ep_from)
        self._ep_starts = [self._ep_from[i] for i in self._ep_order]

        self.delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps is not None:
            self.delta_indices = self._build_delta_indices(delta_timestamps)

        # 懒加载缓存
        self._data_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._frame_cache: dict[tuple[str, float], torch.Tensor] = {}

    # ------------------------------------------------------------------ meta
    def _load_stats(self) -> dict[str, dict]:
        path = self.root / "meta" / "stats.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_tasks(self) -> dict[int, str]:
        import pyarrow.parquet as pq

        path = self.root / "meta" / "tasks.parquet"
        if not path.exists():
            return {}
        table = pq.read_table(path)
        # pandas 写出来是两列：task_index + 索引列（列名可能是 "task" 或 "__index_level_0__"）
        text_col = next(c for c in table.column_names if c != "task_index")
        names = table.column(text_col).to_pylist()
        indices = table.column("task_index").to_pylist()
        return {int(i): str(n) for n, i in zip(names, indices, strict=True)}

    def _load_episodes(self) -> list[dict]:
        import pyarrow.parquet as pq

        # episodes 可能按 chunk 分文件；统一读成 list[dict]，按 episode_index 排序
        rows: list[dict] = []
        for path in sorted((self.root / "meta" / "episodes").glob("*/*.parquet")):
            table = pq.read_table(path)
            rows.extend(table.to_pylist())
        rows.sort(key=lambda r: int(r["episode_index"]))
        if len(rows) != self.total_episodes:
            logger.warning("meta says %d episodes, found %d rows", self.total_episodes, len(rows))
        return rows

    def _build_delta_indices(self, delta_timestamps: dict[str, list[float]]) -> dict[str, list[int]]:
        delta_indices = {}
        for key, deltas in delta_timestamps.items():
            if key not in self.features:
                raise KeyError(f"delta_timestamps refers to unknown feature {key!r}")
            for d in deltas:
                if abs(d * self.fps - round(d * self.fps)) / self.fps > self.tolerance_s:
                    raise ValueError(f"delta {d} for {key} is not a multiple of 1/{self.fps}s")
            delta_indices[key] = [round(d * self.fps) for d in deltas]
        return delta_indices

    # ------------------------------------------------------------------ paths
    def _episode(self, ep_idx: int) -> dict:
        return self.episodes[ep_idx]

    def data_file_path(self, ep_idx: int) -> Path:
        ep = self._episode(ep_idx)
        return self.root / self.data_path_template.format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )

    def video_file_path(self, ep_idx: int, video_key: str) -> Path:
        ep = self._episode(ep_idx)
        return self.root / self.video_path_template.format(
            video_key=video_key,
            chunk_index=ep[f"videos/{video_key}/chunk_index"],
            file_index=ep[f"videos/{video_key}/file_index"],
        )

    # ------------------------------------------------------------------ data
    def _data_file(self, ep_idx: int) -> dict[str, Any]:
        """按 (chunk, file) 缓存 pyarrow 列，并建立 global index -> 行号 的映射。"""
        ep = self._episode(ep_idx)
        key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        if key in self._data_cache:
            return self._data_cache[key]

        import pyarrow.parquet as pq

        table = pq.read_table(self.data_file_path(ep_idx))
        index_col = np.asarray(table.column("index").to_pylist())
        entry = {
            "table": table,
            "row_of_index": {int(v): i for i, v in enumerate(index_col)},
            "index_col": index_col,
            "timestamps": np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64),
        }
        self._data_cache[key] = entry
        return entry

    def get_row(self, abs_index: int) -> dict[str, torch.Tensor]:
        """按全局帧索引取一行（所有表格列都转成 tensor，形状与 features 一致）。"""
        # 找到包含该帧的 episode（dataset_from_index 有序）
        ep_idx = self._episode_of_index(abs_index)
        entry = self._data_file(ep_idx)
        row = entry["row_of_index"][abs_index]
        table = entry["table"]
        item: dict[str, torch.Tensor] = {}
        for key in table.column_names:
            if key in self.video_keys:
                continue
            value = table.column(key)[row].as_py()
            if isinstance(value, (list, tuple)):
                item[key] = torch.tensor(value)
            else:
                item[key] = torch.tensor([value])
        return item

    def _episode_of_index(self, abs_index: int) -> int:
        import bisect

        pos = bisect.bisect_right(self._ep_starts, abs_index) - 1
        ep_idx = self._ep_order[max(pos, 0)]
        if abs_index >= self._ep_to[ep_idx] or abs_index < self._ep_from[ep_idx]:
            raise IndexError(f"index {abs_index} is outside episode {ep_idx}")
        return ep_idx

    def _query_indices(self, abs_idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        assert self.delta_indices is not None
        ep_start = self._ep_from[ep_idx]
        ep_end = self._ep_to[ep_idx]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in deltas]
            for key, deltas in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in deltas]
            )
            for key, deltas in self.delta_indices.items()
        }
        return query_indices, padding

    # ------------------------------------------------------------------ video
    def _decode_frames(self, path: Path, timestamps: list[float]) -> torch.Tensor:
        if self.video_backend != "pyav":
            raise ValueError(f"only the pyav backend is implemented, got {self.video_backend!r}")

        cache_keys = [(str(path), round(ts, 6)) for ts in timestamps]
        if all(k in self._frame_cache for k in cache_keys):
            frames = torch.stack([self._frame_cache[k] for k in cache_keys])
        else:
            frames = decode_video_frames_pyav(
                path, timestamps, self.tolerance_s, return_uint8=self.return_uint8
            )
            for key, frame in zip(cache_keys, frames, strict=True):
                self._frame_cache[key] = frame
        return frames

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        out = {}
        for vid_key, ts in query_timestamps.items():
            from_timestamp = float(self._episode(ep_idx)[f"videos/{vid_key}/from_timestamp"])
            frames = self._decode_frames(self.video_file_path(ep_idx, vid_key), [from_timestamp + t for t in ts])
            out[vid_key] = frames.squeeze(0) if len(ts) == 1 else frames
        return out

    def _query_timestamps(
        self, current_ts: float, query_indices: dict[str, list[int]] | None
    ) -> dict[str, list[float]]:
        query_timestamps: dict[str, list[float]] = {}
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                # 视频帧的时间戳用表格里的 timestamp 列，而不是 idx/fps
                query_timestamps[key] = [
                    float(self.timestamp_of(idx)) for idx in query_indices[key]
                ]
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def timestamp_of(self, abs_index: int) -> float:
        """全局帧索引 -> 时间戳（秒）。"""
        ep_idx = self._episode_of_index(abs_index)
        entry = self._data_file(ep_idx)
        return float(entry["timestamps"][entry["row_of_index"][abs_index]])

    # ------------------------------------------------------------------ item
    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.get_row(idx)
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._query_indices(abs_idx, ep_idx)
            item.update(padding)
            for key, indices in query_indices.items():
                if key in self.video_keys:  # 视频帧由 _query_videos 按时间戳解码
                    continue
                rows = [self.get_row(global_idx) for global_idx in indices]
                item[key] = torch.stack([row[key] for row in rows])

        if self.video_keys:
            current_ts = float(item["timestamp"].item())
            query_timestamps = self._query_timestamps(current_ts, query_indices)
            item = {**self._query_videos(query_timestamps, ep_idx), **item}

        item["task"] = self.tasks.get(int(item["task_index"].item()), "")
        return item

    # ------------------------------------------------------------------ 便捷
    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    def episode_frames(self, ep_idx: int) -> int:
        return self._ep_to[ep_idx] - self._ep_from[ep_idx]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(root={str(self.root)!r}, frames={self.total_frames}, "
            f"episodes={self.total_episodes}, fps={self.fps}, cameras={self.video_keys})"
        )


def decode_video_frames_pyav(
    video_path: str | Path,
    timestamps: list[float],
    tolerance_s: float,
    *,
    return_uint8: bool = False,
) -> torch.Tensor:
    """按时间戳取帧（移植自 lerobot `video_utils.decode_video_frames_pyav`）。

    PyAV 只能 seek 到关键帧，所以先退到 `min(timestamps)` 之前的关键帧，再逐帧解码
    到覆盖 `max(timestamps)` 为止；随后对每个查询时间戳取最近的一帧，并校验误差
    不超过 tolerance_s。
    """
    import av

    video_path = str(video_path)
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(
            round(first_ts / stream.time_base) - 1,
            backward=True,
            any_frame=False,
            stream=stream,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(f"no frames decoded from {video_path} within [{first_ts}, {last_ts}]")

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)
    if (min_ > tolerance_s).any():
        raise FrameTimestampError(
            f"query timestamps drift more than {tolerance_s}s from decoded frames in {video_path}: "
            f"{min_[min_ > tolerance_s].tolist()}"
        )

    frames = torch.stack([loaded_frames[i] for i in argmin_.tolist()])
    if return_uint8:
        return frames
    return frames.to(torch.float32) / 255.0
