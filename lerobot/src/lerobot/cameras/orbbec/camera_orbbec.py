# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the OrbbecCamera class for capturing frames from Orbbec RGB-D cameras.

Unlike the Intel RealSense backend, Orbbec ships two incompatible generations of its Python SDK and
its device families are split across them (e.g. the Astra Pro Plus is only supported by the v1 SDK),
so this module imports the SDK defensively and reports a precise error for an unsupported pairing.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _pyorbbecsdk_available

if TYPE_CHECKING or _pyorbbecsdk_available:
    import pyorbbecsdk as ob
else:
    ob = None

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


def _require_orbbec_sdk() -> None:
    """Raise an actionable ImportError when `pyorbbecsdk` is missing.

    `require_package` cannot be used here because Orbbec publishes no package on PyPI, so its
    generic "pip install 'lerobot[orbbec]'" hint would point at an extra that does not exist.
    """
    if not _pyorbbecsdk_available:
        raise ImportError(
            "'pyorbbecsdk' is required but not installed. Orbbec does not publish it on PyPI: build "
            "it from https://github.com/orbbec/pyorbbecsdk (branch `main` = Orbbec SDK v1.x, the "
            "only branch supporting older devices such as the Astra Pro Plus) and install the "
            "resulting wheel, or run `scripts/install_orbbec_sdk.sh` shipped with this repository."
        )


def _camera_identifier(info: Any, index: int) -> str:
    """Best available identifier for a device: serial number, then UID, name and enumeration index.

    Some devices (e.g. an Astra Pro Plus on a USB 2.0 link) report an empty serial number, which
    would otherwise produce an unusable empty `id` that cannot be matched back to a device.
    """
    for candidate in (info.get_serial_number(), info.get_uid(), info.get_name()):
        if candidate:
            return str(candidate)
    return f"index:{index}"


def _format_name(fmt: Any) -> str:
    """Normalizes an `OBFormat` to its bare name, e.g. `"MJPG"` or `"Y12"`.

    This indirection is mandatory: `str(ob.OBFormat.Y12)` is `"OBFormat.Y12"`, *not* `"Y12"`, so
    comparing it against the bare names in `_PREFERRED_COLOR_FORMATS` / `_DEPTH_16BIT_FORMATS`
    matches nothing and rejects every single frame. `OBFormat.name` is preferred because it is
    stable regardless of how the SDK formats the enum for display.
    """
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(fmt)
    return text.rsplit(".", 1)[-1] if "." in text else text


def _default_video_profile(device: Any, sensor_type: Any) -> dict[str, Any] | None:
    """Returns the width/height/fps/format of a sensor's default video profile, or None."""
    sensor = device.get_sensor_list().get_sensor_by_type(sensor_type)
    if sensor is None:
        return None

    profile_list = sensor.get_stream_profile_list()
    if profile_list.get_count() == 0:
        return None

    profile = profile_list.get_default_video_stream_profile()
    return {
        "format": _format_name(profile.get_format()),
        "width": profile.get_width(),
        "height": profile.get_height(),
        "fps": profile.get_fps(),
    }


# Formats a color stream profile may advertise. Ordered by preference: packed RGB/BGR frames need
# no decoding, MJPG costs a JPEG decode but is what most UVC-based Orbbec devices stream by default.
_PREFERRED_COLOR_FORMATS = ("RGB", "BGR", "MJPG", "YUYV", "YUY2", "UYVY", "NV12", "NV21", "I420", "YV12")

# 16-bit containers used by Orbbec depth streams (Y11/Y12/Y14 pack fewer bits into a 16-bit word).
_DEPTH_16BIT_FORMATS = ("Y16", "Z16", "Y14", "Y12", "Y11", "Y10", "RW16", "DISP16", "GRAY")

# How long the background thread waits for a frame set before looping back to the stop event.
_READ_TIMEOUT_MS = 1000

# Upper bound for blocking read()/read_depth() calls.
_BLOCKING_READ_TIMEOUT_MS = 10000

# `connect(warmup=True)` always reads for at least this long, even if frames are already there.
_MIN_WARMUP_S = 1.0


class OrbbecCamera(Camera):
    """
    Manages interactions with Orbbec RGB-D cameras for frame and depth recording.

    This class provides an interface similar to `OpenCVCamera`/`RealSenseCamera` but tailored for
    Orbbec devices, leveraging the `pyorbbecsdk` library. Devices are identified by their serial
    number or by their human-readable name; a single-camera setup can rely on auto-selection.

    Depth frames are returned as `np.uint16` of shape `(H, W, 1)` holding millimetres, exactly like
    `RealSenseCamera`, so the rest of LeRobot (dataset features, video encoding, the `DepthCamera`
    protocol) works unchanged. The device's own depth scale is applied for you.

    Use the provided utility script to find available devices and their default profiles:
    ```bash
    lerobot-find-cameras orbbec
    ```

    Example:
        ```python
        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Auto-select the only connected device, color + depth (defaults)
        camera = OrbbecCamera(OrbbecCameraConfig())
        camera.connect()

        color_image = camera.read()               # (H, W, 3) uint8, RGB
        depth_map = camera.read_depth()           # (H, W, 1) uint16, millimetres

        # Pinned resolution/format, color only, rotated
        config = OrbbecCameraConfig(
            serial_number_or_name="0123456789",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            use_depth=False,
            rotation=Cv2Rotation.ROTATE_180,
        )
        with OrbbecCamera(config) as cam:
            frame = cam.read()
        ```
    """

    def __init__(self, config: OrbbecCameraConfig):
        """
        Initializes the OrbbecCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        _require_orbbec_sdk()
        super().__init__(config)
        self.config = config

        self.width: int | None = config.width
        self.height: int | None = config.height
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.align_depth_to_color = config.align_depth_to_color

        self.requested_identifier: str | None = config.serial_number_or_name
        self.serial_number: str | None = None
        self.device_name: str | None = None

        self.context: Any | None = None
        self.device: Any | None = None
        self.pipeline: Any | None = None
        # Only ever assigned so that the SDK `Config` outlives `pipeline.start()`.
        self.config_obj: Any | None = None
        self.align_filter: Any | None = None
        self.color_profile: Any | None = None
        self.depth_profile: Any | None = None

        self.color_format: str | None = None
        self.depth_format: str | None = None
        self.depth_scale: float = 1.0

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = None
        self.capture_height: int | None = None
        # Raw size of the depth stream. Equal to the color size while D2C alignment is on.
        self.depth_capture_width: int | None = None
        self.depth_capture_height: int | None = None
        self._reset_connection_settings()

    def __str__(self) -> str:
        identifier = self.serial_number or self.device_name or self.requested_identifier or "auto"
        return f"{self.__class__.__name__}({identifier})"

    def _reset_connection_settings(self) -> None:
        """Restore settings that may have been auto-detected during a failed connection."""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.warmup_s = self.config.warmup_s
        # `capture_width`/`capture_height` always hold the *raw* sensor frame, which is the input
        # validated by `_postprocess_image`. Rotation only affects the output size (`width`/`height`).
        self.capture_width, self.capture_height = self.width, self.height
        self.depth_capture_width, self.depth_capture_height = self.width, self.height

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.pipeline is not None

    def _open_pipeline(self) -> None:
        """Initializes the Orbbec pipeline, starts it, and starts the background read thread.

        Raises:
            ConnectionError: If no device matches the configuration, or the pipeline fails to start.
            RuntimeError: If the device does not expose a stream that was requested.
        """
        self._select_device()

        pipeline = ob.Pipeline(self.device)
        config = ob.Config()
        self._configure_pipeline_config(config)

        try:
            pipeline.start(config)
        except Exception as e:
            raise ConnectionError(
                f"{self}: failed to start the Orbbec pipeline. Run `lerobot-find-cameras orbbec` "
                "to find available cameras."
            ) from e

        self.pipeline = pipeline
        self.config_obj = config

        # Software D2C alignment makes the depth map share the color frame's field of view, which is
        # what LeRobot's `*_depth` sibling features assume. Only meaningful with both streams on.
        if self.align_depth_to_color and self.use_rgb and self.use_depth:
            self.align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            logger.info(f"{self}: depth frames will be aligned to the color stream (software D2C).")

        try:
            self._configure_capture_settings()
            self._start_read_thread()
        except BaseException:
            self._release_after_failed_setup()
            raise

    def _configure_pipeline_config(self, config: Any) -> None:
        """Creates and configures the Orbbec pipeline configuration object.

        Raises:
            RuntimeError: If the device does not expose a requested stream.
        """
        enabled: list[str] = []

        if self.use_rgb:
            color_profile = self._select_stream_profile(
                ob.OBSensorType.COLOR_SENSOR, "color", _PREFERRED_COLOR_FORMATS
            )
            if color_profile is None:
                raise RuntimeError(f"{self}: the device exposes no color stream profile.")
            config.enable_stream(color_profile)
            self.color_profile = color_profile
            self.color_format = _format_name(color_profile.get_format())
            enabled.append(f"color={self.color_format}")

        if self.use_depth:
            depth_profile = self._select_stream_profile(
                ob.OBSensorType.DEPTH_SENSOR, "depth", _DEPTH_16BIT_FORMATS
            )
            if depth_profile is None:
                raise RuntimeError(f"{self}: the device exposes no depth stream profile.")
            config.enable_stream(depth_profile)
            self.depth_profile = depth_profile
            self.depth_format = _format_name(depth_profile.get_format())
            enabled.append(f"depth={self.depth_format}")

        logger.info(f"{self}: pipeline configured ({', '.join(enabled)}).")

    def _run_warmup(self) -> None:
        """Blocks until at least one frame from every enabled stream has been captured.

        `warmup_s` is only a lower bound: an Orbbec depth stream can need a moment longer than the
        color stream before its first frame arrives, and that is a slow start, not a fault.

        Raises:
            ConnectionError: If an enabled stream had produced no frame when warmup ended.
        """
        self.warmup_s = max(self.warmup_s, _MIN_WARMUP_S)

        warmup_read = self.async_read if self.use_rgb else self.async_read_depth
        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            try:
                warmup_read(timeout_ms=self.warmup_s * 1000)
            except (TimeoutError, RuntimeError) as e:
                # The two Orbbec streams start (and deliver their first frame) independently, and
                # they share one frame event. The depth stream typically reaches `STREAMING` a few
                # hundred ms before the color stream, so a depth frame notifies a color read while
                # `latest_color_frame` is still empty, which that read reports as an error. That is
                # a slow start of the *other* stream, not a fault: keep polling for the full budget
                # and let the check below decide whether a stream really stayed silent.
                logger.debug(f"{self}: warmup read is not ready yet ({e}).")
            time.sleep(0.1)

        with self.frame_lock:
            missing = [
                label
                for label, enabled, frame in (
                    ("color", self.use_rgb, self.latest_color_frame),
                    ("depth", self.use_depth, self.latest_depth_frame),
                )
                if enabled and frame is None
            ]

        if missing:
            raise ConnectionError(f"{self} received no {' and no '.join(missing)} frame during warmup.")

    def _release_after_failed_setup(self) -> None:
        """Releases the device handle and restores auto-detected settings after a failed attempt."""
        try:
            self._cleanup_resources()
        except Exception:
            logger.exception(f"Failed to fully clean up {self} after connect() failed.")
        self._reset_connection_settings()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the Orbbec camera specified in the configuration.

        Resolves the device, enables the color (and optionally depth) streams, starts the pipeline
        and launches the background read thread. When `warmup` is True it then waits until every
        enabled stream has produced a frame.

        Args:
            warmup (bool): If True, waits at connect() time until every enabled stream has produced
                a frame. Defaults to True.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration matches no device or more than one.
            ConnectionError: If no device is detected, the pipeline fails to start, or an enabled
                stream produces no frame during warmup.
            RuntimeError: If the device does not expose a requested stream.
        """
        self._open_pipeline()

        if warmup:
            try:
                self._run_warmup()
            except BaseException:
                self._release_after_failed_setup()
                raise

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains 'type',
            'id' (serial number), 'name', uid, pid/vid, connection type, firmware version and the
            default stream profiles reported by the color and depth sensors.

        Raises:
            ImportError: If `pyorbbecsdk` is not installed.
        """
        _require_orbbec_sdk()

        found_cameras_info: list[dict[str, Any]] = []
        context = ob.Context()
        device_list = context.query_devices()

        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()

            camera_info: dict[str, Any] = {
                "name": info.get_name(),
                "type": "Orbbec",
                "id": _camera_identifier(info, index),
                "serial_number": info.get_serial_number(),
                "uid": info.get_uid(),
                "pid": hex(info.get_pid()),
                "vid": hex(info.get_vid()),
                "connection_type": info.get_connection_type(),
                "firmware_version": info.get_firmware_version(),
            }

            for label, sensor_type in (
                ("color", ob.OBSensorType.COLOR_SENSOR),
                ("depth", ob.OBSensorType.DEPTH_SENSOR),
            ):
                profile = _default_video_profile(device, sensor_type)
                if profile is not None:
                    camera_info[f"default_{label}_stream_profile"] = profile

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _select_device(self) -> Any:
        """Resolves the configured serial/name into a concrete Orbbec `Device`."""
        self.context = ob.Context()
        device_list = self.context.query_devices()
        count = device_list.get_count()

        if count == 0:
            raise ConnectionError(
                f"{self}: no Orbbec camera detected. Run `lerobot-find-cameras orbbec` to list "
                "available devices (on Linux, a non-root user also needs the Orbbec udev rules)."
            )

        infos = []
        for index in range(count):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            infos.append((device, info, _camera_identifier(info, index)))

        # An empty string means "not configured" just like None, which keeps `serial_number_or_name=""`
        # (what a caller gets from a device with no serial number) from being treated as a real filter.
        requested = self.requested_identifier or None
        if requested is None:
            device, info, _ = infos[0]
            if count > 1:
                logger.warning(
                    f"{self}: {count} Orbbec cameras detected and no `serial_number_or_name` was "
                    f"configured; using the first one ({info.get_name()}). Set the serial "
                    "number explicitly to remove the ambiguity."
                )
        else:
            matches = [
                (device, info)
                for device, info, identifier in infos
                if requested in (identifier, info.get_serial_number(), info.get_uid(), info.get_name())
            ]
            if not matches:
                available = [f"{info.get_name()} (id={identifier})" for _, info, identifier in infos]
                raise ValueError(
                    f"{self}: no Orbbec camera matches '{requested}'. Available devices: {available}"
                )
            if len(matches) > 1:
                serial_numbers = [info.get_serial_number() for _, info in matches]
                raise ValueError(
                    f"{self}: multiple Orbbec cameras match '{requested}'. "
                    f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
                )
            device, info = matches[0]

        self.serial_number = info.get_serial_number()
        self.device_name = info.get_name()
        self.device = device
        return device

    def _select_stream_profile(
        self,
        sensor_type: Any,
        sensor_label: str,
        preferred_formats: tuple[str, ...],
    ) -> Any:
        """
        Picks the stream profile to enable for a sensor, or None when it exposes none.

        When `fps`/`width`/`height` are configured the exact mode is matched, preferring the
        formats in the given order. Otherwise the sensor's own default profile is used, so the
        camera behaves sensibly with a zero-effort config.
        """
        sensor = self.device.get_sensor_list().get_sensor_by_type(sensor_type)
        if sensor is None:
            return None

        profile_list = sensor.get_stream_profile_list()
        video_profiles = [
            profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
            for index in range(profile_list.get_count())
            if profile_list.get_stream_profile_by_index(index).is_video_stream_profile()
        ]
        if not video_profiles:
            return None

        if self.width and self.height and self.fps:
            requested_mode = (self.width, self.height, self.fps)
            for fmt_name in preferred_formats:
                for profile in video_profiles:
                    mode = (profile.get_width(), profile.get_height(), profile.get_fps())
                    if _format_name(profile.get_format()) == fmt_name and mode == requested_mode:
                        logger.info(
                            f"{self}: using {sensor_label} profile "
                            f"{self.width}x{self.height}@{self.fps} {fmt_name}."
                        )
                        return profile

            logger.warning(
                f"{self}: no {sensor_label} profile matches "
                f"{self.width}x{self.height}@{self.fps}; falling back to the sensor default."
            )

        return profile_list.get_default_video_stream_profile()

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """Reads the enabled stream profiles back and stores the actual mode and capture sizes.

        The enabled profiles - not the requested ones - are authoritative: a sensor may fall back
        to a different mode, and `width`/`height` must describe what is really streamed, otherwise
        `_postprocess_image` rejects every frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If no stream profile was enabled.
        """
        primary = self.color_profile if self.color_profile is not None else self.depth_profile
        if primary is None:
            raise RuntimeError(f"{self}: no stream profile is enabled before use.")

        self.fps = primary.get_fps()

        actual_width = int(round(primary.get_width()))
        actual_height = int(round(primary.get_height()))
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = actual_height, actual_width
        else:
            self.width, self.height = actual_width, actual_height

        # The raw capture size never depends on rotation; both streams are rotated later, in
        # `_postprocess_image`, so only `width`/`height` above track the rotated output size.
        self.capture_width, self.capture_height = actual_width, actual_height

        # Depth keeps its own resolution unless the software align filter resamples it onto color.
        if self.align_filter is not None or self.depth_profile is None:
            self.depth_capture_width, self.depth_capture_height = (
                self.capture_width,
                self.capture_height,
            )
        else:
            self.depth_capture_width = int(round(self.depth_profile.get_width()))
            self.depth_capture_height = int(round(self.depth_profile.get_height()))

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for `read`/`read_depth`: wait for a fresh color or depth frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self._async_read(timeout_ms=_BLOCKING_READ_TIMEOUT_MS, read_depth=read_depth)

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single depth frame synchronously from the camera.

        This is a blocking call. It waits for the camera hardware to deliver a fresh frame set.

        Returns:
            np.ndarray: The depth map as a NumPy array `(H, W, 1)` of type `np.uint16`, holding
            the distance from the sensor in millimetres.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera was configured with `use_depth=False`, or frames are invalid.
        """
        if timeout_ms:
            logger.warning(
                f"{self} read_depth() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._read(read_depth=True)

    def _read_from_hardware(self) -> Any:
        """Waits for one frame set from the pipeline and applies D2C alignment when enabled."""
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        frames = self.pipeline.wait_for_frames(_READ_TIMEOUT_MS)
        if frames is None:
            return None

        if self.align_filter is not None:
            # The software D2C filter needs both streams: given a color-only frame set it logs
            # "pFrame is nullptr!" for every frame and returns nothing usable. Only align complete
            # sets - which is what Orbbec's own examples do - and let an incomplete set through
            # untouched, so a stalled depth stream degrades to "no depth" rather than "no frames".
            has_both_streams = frames.get_color_frame() is not None and frames.get_depth_frame() is not None
            if has_both_streams:
                aligned_frames = self.align_filter.process(frames)
                if aligned_frames is not None:
                    frames = aligned_frames

        return frames

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """
        Reads a single color frame synchronously from the camera.

        This is a blocking call. It waits for the camera hardware to deliver a fresh frame set.

        Returns:
            np.ndarray: The captured color frame as a NumPy array `(H, W, 3)`, processed according
            to `color_mode` and `rotation`.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera was configured with `use_rgb=False`, or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._read()

    def _color_to_array(self, frame: Any) -> NDArray[Any]:
        """Decodes a color frame into an owned `(H, W, 3)` uint8 RGB/BGR array."""
        fmt = _format_name(frame.get_format())
        width = frame.get_width()
        height = frame.get_height()
        data = frame.get_data()

        if fmt == "RGB":
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if self.color_mode == ColorMode.BGR:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif fmt == "BGR":
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if self.color_mode == ColorMode.RGB:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif fmt in ("BGRA", "RGBA"):
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            array = cv2.cvtColor(
                array,
                cv2.COLOR_BGRA2BGR if fmt == "BGRA" else cv2.COLOR_RGBA2BGR,
            )
            if self.color_mode == ColorMode.RGB:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif fmt == "MJPG":
            array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if array is None:
                raise RuntimeError(f"{self}: failed to decode the MJPG color frame.")
            if self.color_mode == ColorMode.RGB:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif fmt in ("YUYV", "YUY2"):
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
            array = cv2.cvtColor(
                array,
                cv2.COLOR_YUV2RGB_YUYV if self.color_mode == ColorMode.RGB else cv2.COLOR_YUV2BGR_YUYV,
            )
        elif fmt == "UYVY":
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
            array = cv2.cvtColor(
                array,
                cv2.COLOR_YUV2RGB_UYVY if self.color_mode == ColorMode.RGB else cv2.COLOR_YUV2BGR_UYVY,
            )
        elif fmt in ("NV12", "NV21"):
            array = np.frombuffer(data, dtype=np.uint8).reshape(height * 3 // 2, width)
            array = cv2.cvtColor(
                array,
                cv2.COLOR_YUV2RGB_NV12 if fmt == "NV12" else cv2.COLOR_YUV2RGB_NV21,
            )
            if self.color_mode == ColorMode.BGR:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif fmt in ("I420", "YV12"):
            array = np.frombuffer(data, dtype=np.uint8).reshape(height * 3 // 2, width)
            array = cv2.cvtColor(
                array,
                cv2.COLOR_YUV2RGB_I420 if fmt == "I420" else cv2.COLOR_YUV2RGB_YV12,
            )
            if self.color_mode == ColorMode.BGR:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif fmt in ("Y8", "GRAY"):
            gray = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
            array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if self.color_mode == ColorMode.BGR:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                f"{self}: unsupported Orbbec color format '{fmt}'. Supported formats: "
                f"{', '.join(_PREFERRED_COLOR_FORMATS)}."
            )

        return np.ascontiguousarray(array)

    def _depth_to_array(self, frame: Any) -> NDArray[Any]:
        """Converts a depth frame into an owned `(H, W)` uint16 array of millimetres."""
        fmt = _format_name(frame.get_format())
        if fmt not in _DEPTH_16BIT_FORMATS:
            raise ValueError(
                f"{self}: unsupported Orbbec depth format '{fmt}'. Expected one of: "
                f"{', '.join(_DEPTH_16BIT_FORMATS)}."
            )

        width = frame.get_width()
        height = frame.get_height()
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width)

        scale = float(frame.get_depth_scale())
        # A device that reports no scale (0.0) already delivers millimetres, so multiplying by it
        # would silently zero out the whole depth map.
        if np.isfinite(scale) and scale > 0.0 and scale != 1.0:
            scaled = np.rint(raw.astype(np.float32) * scale)
            raw = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            raw = raw.copy()

        self.depth_scale = scale
        return raw

    def _postprocess_image(self, image: NDArray[Any], depth_frame: bool = False) -> NDArray[Any]:
        """
        Validates dimensions and applies rotation to a raw color or depth frame.

        Args:
            image: The frame, `(H, W, 3)` for color and `(H, W)` for depth.
            depth_frame: Whether `image` is a depth map.

        Returns:
            The rotated frame.

        Raises:
            RuntimeError: If the raw frame dimensions do not match the configured capture size.
        """
        expected_width = self.depth_capture_width if depth_frame else self.capture_width
        expected_height = self.depth_capture_height if depth_frame else self.capture_height

        height, width = image.shape[:2]
        if not depth_frame and image.shape[2] != 3:
            raise RuntimeError(
                f"{self} frame channels={image.shape[2]} do not match expected 3 channels (RGB/BGR)."
            )

        if (
            expected_width is not None
            and expected_height is not None
            and (width != expected_width or height != expected_height)
        ):
            raise RuntimeError(
                f"{self} frame width={width} or height={height} do not match configured "
                f"width={expected_width} or height={expected_height}."
            )

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            return cv2.rotate(image, self.rotation)
        return image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration it fetches a frame set, stores the decoded color/depth frames under the
        frame lock and notifies listeners through `new_frame_event`.
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                frames = self._read_from_hardware()
                if frames is None:
                    continue

                processed_color_frame = None
                processed_depth_frame = None

                if self.use_rgb:
                    color_frame = frames.get_color_frame()
                    if color_frame is not None:
                        processed_color_frame = self._postprocess_image(self._color_to_array(color_frame))

                if self.use_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is not None:
                        depth = self._postprocess_image(self._depth_to_array(depth_frame), depth_frame=True)
                        if depth.ndim == 2:  # (H, W) -> (H, W, 1)
                            depth = depth[..., np.newaxis]
                        processed_depth_frame = depth

                if processed_color_frame is None and processed_depth_frame is None:
                    continue

                capture_time = time.perf_counter()

                with self.frame_lock:
                    # Under the lock, so a late frame cannot resurrect the buffer _stop_read_thread() cleared.
                    if stop_event.is_set():
                        break
                    if processed_color_frame is not None:
                        self.latest_color_frame = processed_color_frame
                    if processed_depth_frame is not None:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():  # pragma: no cover
                logger.warning(f"{self} read thread did not terminate within timeout.")

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _cleanup_resources(self) -> None:
        """Stop background reads and stop the pipeline, including after partial setup."""
        read_thread = self.thread
        pipeline = self.pipeline

        try:
            self._stop_read_thread()
        finally:
            self.pipeline = None
            self.config_obj = None
            self.align_filter = None
            self.color_profile = None
            self.depth_profile = None
            self.color_format = None
            self.depth_format = None
            try:
                if pipeline is not None:
                    pipeline.stop()
            finally:
                # Stopping the pipeline may unblock a hardware read that outlived
                # the first bounded join in _stop_read_thread().
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2.0)
                    if read_thread.is_alive():  # pragma: no cover
                        logger.warning(f"{self} read thread remained alive after stopping the pipeline.")

    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for `async_read`/`async_read_depth`: return the latest buffered frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background read thread.
        It does not block waiting for the camera hardware directly, but may wait up to `timeout_ms`
        for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame to become
                available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured color frame, processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If `use_rgb` is False or the background thread died unexpectedly.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._async_read(timeout_ms=timeout_ms)

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for `read_latest`/`read_latest_depth`: peek the latest buffered frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent (color) frame captured immediately (peeking).

        This method is non-blocking and returns whatever is currently in the memory buffer. The
        frame may be stale, meaning it could have been captured a while ago.

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._read_latest(max_age_ms=max_age_ms)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """Read the latest depth frame asynchronously, in millimetres.

        Mirrors `async_read` but returns the depth stream rather than the color stream. Output is
        `np.uint16` of shape `(H, W, 1)`, where each pixel is the distance from the sensor in
        millimetres.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If `use_depth` is False for this camera, or the read thread is not running.
            TimeoutError: If no frame becomes available within `timeout_ms`.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent depth frame in millimetres (peeking).

        Non-blocking counterpart of `read_latest` for the depth stream. Output is `np.uint16` of
        shape `(H, W, 1)`, where each pixel is the distance from the sensor in millimetres.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If `use_depth` is False, or no depth frame has been captured yet.
            TimeoutError: If the latest depth frame is older than `max_age_ms`.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        self._cleanup_resources()
        logger.info(f"{self} disconnected.")
