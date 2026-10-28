# dataset —— LeRobotDataset v3.0 读取

`lerobot_v3.py` 是一个最小实现：只用 `pyarrow`（parquet）+ `PyAV`（视频）+ `torch`，
**不 import lerobot，也不 import huggingface 的 `datasets`**，可以直接对着一份
下载好的 v3 数据集目录工作。

## v3 布局（`meta/info.json` 的 `codebase_version` 以 `v3` 开头）

```
<root>/
  meta/info.json                     # features（dtype/shape/names）、fps、文件模板、total_frames
  meta/stats.json                    # 每个 feature 的 min/max/mean/std/count
  meta/tasks.parquet                 # task_index -> 任务文本
  meta/episodes/chunk-000/file-000.parquet
                                     # 每集一行：数据/视频分片位置、全局帧索引区间、
                                     # 视频时间偏移（from_timestamp）、每集统计量
  data/chunk-000/file-000.parquet    # 帧级表格：observation.state / action / index / timestamp …
  videos/<video_key>/chunk-000/file-000.mp4
```

与 v2.1 的区别（这也是为什么 v2.1 需要转换）：v2.1 是"每集一个 parquet + 每集一个
mp4 + `meta/episodes.jsonl`"，v3 把多集打进同一个（可分片的）数据文件与视频文件，
episode 的定位信息改放在 `meta/episodes/*.parquet` 里（`dataset_from_index` /
`dataset_to_index` / `videos/.../from_timestamp`）。

## 用法

```python
from dataset import LeRobotV3Dataset

# ACT 训练/推理需要的动作块：从当前帧起 100 帧
delta_timestamps = {"action": [i / 50 for i in range(100)]}
ds = LeRobotV3Dataset("dataset/data/aloha_sim_insertion_human", delta_timestamps=delta_timestamps)

item = ds[0]
item["observation.images.top"]   # (3, 480, 640) float32 in [0,1]
item["observation.state"]        # (14,)
item["action"]                   # (100, 14)，越界处被夹到 episode 末尾
item["action_is_pad"]            # (100,) bool，True 表示该步是补齐出来的
item["task"]                     # "Insert the peg into the socket."
```

`__getitem__` 的返回语义与 lerobot 0.6.2 的 `DatasetReader.get_item()` 对齐：

* 索引是数据集内的**行号**（未做 episode 过滤时等于 `index` 列）；
* `delta_timestamps` 换算成帧偏移后**夹到 episode 边界**，并给出 `<key>_is_pad`；
* 视频取帧：查询时间戳 = 该帧 `timestamp` + 该集的 `videos/.../from_timestamp`，
  取最近帧，误差超过 `tolerance_s`（默认 1e-4s）直接抛 `FrameTimestampError`；
* 单相机返回 `(C, H, W)`，多帧（例如给了图像 delta）返回 `(T, C, H, W)`。

其它 API：`ds.stats`（norm 统计量）、`ds.tasks`、`ds.episodes`、
`ds.episode_frames(i)`、`ds.timestamp_of(index)`。

## 关于格式转换

本仓库用到的两个示例数据集都已经是 v3.0，不需要转换脚本：

| 数据集 | 规模 | 相机键 | 用途 |
| --- | --- | --- | --- |
| `lerobot/aloha_sim_insertion_human` | 50 集 / 25000 帧 / 50fps | `observation.images.top` 480×640 | `policy/ACT` |
| `lerobot/pusht` | 206 集 / 25650 帧 / 10fps | `observation.image` 96×96 | `policy/DP` |

`LeRobotV3Dataset` 遇到非 v3 版本会直接报错并提示
`python -m lerobot.scripts.convert_dataset_v21_to_v30`。

注意 pusht 的相机是**单数键** `observation.image`（不是 `observation.images.*`），
读取器按 `features[*].dtype == "video"` 判断要解码哪些列，不受命名影响；
视频键也可以给 `delta_timestamps`（DP 的 `n_obs_steps=2` 就需要历史帧），
此时返回 `(T, C, H, W)`。

## 性能与缓存

* 数据分片按 `(chunk_index, file_index)` 懒加载并缓存，每个分片建一份
  `全局 index -> 行号` 映射（25k 行约 1MB 内存）；
* 视频帧按 `(视频路径, 时间戳)` 做 LRU 之外的最小缓存（`_frame_cache`），
  逐 step 推理重复访问同一帧时不会二次解码；
* 视频用 PyAV seek 到关键帧后顺序解码到目标时间戳，单帧查询在 AV1 480×640 上
  约 10~30ms/帧（关键帧间隔决定）。

## 下载

本目录下的数据是手工用 `curl` 拉的：`huggingface_hub.snapshot_download` 在本机
（SOCKS 代理 + Xet 后端）会卡在大文件上，而 `hf-mirror.com` 对 dataset 的
`resolve` 路径返回 404，只有官方域名可用。

```bash
BASE=https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/resolve/main
DEST=dataset/data/aloha_sim_insertion_human
for f in meta/info.json meta/stats.json meta/tasks.parquet \
         meta/episodes/chunk-000/file-000.parquet \
         data/chunk-000/file-000.parquet data/chunk-000/file-001.parquet \
         data/chunk-000/file-002.parquet data/chunk-000/file-003.parquet \
         videos/observation.images.top/chunk-000/file-000.mp4 ; do
  mkdir -p $DEST/$(dirname $f)
  curl -sL -C - -o $DEST/$f $BASE/$f
done
```

注意这个数据集的 50 集**共用同一个视频文件**（`file_index` 全是 0，靠
`from_timestamp` 定位每集），所以只有 `videos/.../file-000.mp4` 存在，约 120MB。

校验完整性：

```bash
python - <<'PY'
import pyarrow.parquet as pq, glob
print(sum(pq.read_table(p).num_rows for p in glob.glob("dataset/data/aloha_sim_insertion_human/data/*/*.parquet")))
PY
# 应输出 25000，与 meta/info.json 的 total_frames 一致
```
