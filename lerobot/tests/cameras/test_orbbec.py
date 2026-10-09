#!/usr/bin/env python

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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_orbbec.py
# ```

from unittest.mock import patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

pytest.importorskip("pyorbbecsdk")

import pyorbbecsdk as ob

from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot.cameras.orbbec.camera_orbbec import _format_name
from lerobot.scripts.lerobot_find_cameras import create_camera_instance

# Default fake stream geometry: VGA, the profile most Orbbec devices expose. The camera is driven
# entirely through a fake `pyorbbecsdk` backend so these tests never touch real hardware.
WIDTH = 640
HEIGHT = 480
FPS = 30
# The genuine SDK enum members rather than bare strings: `str(ob.OBFormat.RGB)` is "OBFormat.RGB",
# not "RGB", which is exactly the trap the format-handling code used to fall into. Driving the
# fakes with the real objects keeps the whole suite honest about it.
COLOR_FORMAT = ob.OBFormat.RGB
DEPTH_FORMAT = ob.OBFormat.Y16
# Depth formats the fake backend knows how to lay out; every other format becomes a color frame.
DEPTH_FORMATS = (ob.OBFormat.Y16, ob.OBFormat.Y12, ob.OBFormat.Y11, ob.OBFormat.Z16)
SERIAL_NUMBER = "0123456789"
DEVICE_NAME = "Orbbec Astra Pro Plus"
DEPTH_FILL_MM = 1000


class FakeVideoStreamProfile:
    """Minimal stand-in for `ob.VideoStreamProfile`."""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS, fmt=COLOR_FORMAT):
        self._width = width
        self._height = height
        self._fps = fps
        self._fmt = fmt

    def is_video_stream_profile(self):
        return True

    def as_video_stream_profile(self):
        return self

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_fps(self):
        return self._fps

    def get_format(self):
        return self._fmt


class FakeStreamProfileList:
    """Minimal stand-in for `ob.StreamProfileList`."""

    def __init__(self, profiles):
        self._profiles = list(profiles)

    def get_count(self):
        return len(self._profiles)

    def get_stream_profile_by_index(self, index):
        return self._profiles[index]

    def get_default_video_stream_profile(self):
        # The real SDK raises on an empty list (`CHECK_NULLPTR` after `getProfile(0)`), so callers
        # must guard on the count rather than catch. Mirroring that here keeps the fake honest.
        if not self._profiles:
            raise RuntimeError("stream profile list is empty")
        return self._profiles[0]


class FakeFrame:
    """Duck-typed video/depth frame returning fixed data, so no camera is required."""

    def __init__(self, width, height, fmt, depth_scale=1.0, fill=0):
        self._width = width
        self._height = height
        self._fmt = fmt
        self._depth_scale = depth_scale
        if fmt in DEPTH_FORMATS:
            data = np.full((height, width), fill, dtype=np.uint16)
        else:
            data = np.full((height, width, 3), fill, dtype=np.uint8)
        self._data = np.ascontiguousarray(data)

    def get_data(self):
        return self._data

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_format(self):
        return self._fmt

    def get_depth_scale(self):
        return self._depth_scale


class FakeFrameSet:
    """Minimal stand-in for `ob.FrameSet`."""

    def __init__(self, color=None, depth=None):
        self._color = color
        self._depth = depth

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class FakeDeviceInfo:
    """Minimal stand-in for `ob.DeviceInfo`."""

    def __init__(self, serial=SERIAL_NUMBER, name=DEVICE_NAME, uid="uid-0"):
        self._serial = serial
        self._name = name
        self._uid = uid

    def get_serial_number(self):
        return self._serial

    def get_name(self):
        return self._name

    def get_uid(self):
        return self._uid

    def get_pid(self):
        return 0x060F

    def get_vid(self):
        return 0x2BC5

    def get_connection_type(self):
        return "USB"

    def get_firmware_version(self):
        return "1.0.0"


class FakeSensor:
    def __init__(self, profile_list):
        self._profile_list = profile_list

    def get_stream_profile_list(self):
        return self._profile_list


class FakeSensorList:
    def __init__(self, color_profiles, depth_profiles):
        self._color_profiles = color_profiles
        self._depth_profiles = depth_profiles

    def get_sensor_by_type(self, sensor_type):
        if sensor_type == ob.OBSensorType.COLOR_SENSOR:
            profiles = self._color_profiles
        elif sensor_type == ob.OBSensorType.DEPTH_SENSOR:
            profiles = self._depth_profiles
        else:
            return None
        # A device without the sensor reports no list at all, exactly like the SDK's null sensor.
        return None if profiles is None else FakeSensor(profiles)


class FakeDevice:
    def __init__(self, info, color_profiles, depth_profiles):
        self._info = info
        self._sensor_list = FakeSensorList(color_profiles, depth_profiles)

    def get_device_info(self):
        return self._info

    def get_sensor_list(self):
        return self._sensor_list


class FakeDeviceList:
    def __init__(self, devices):
        self._devices = devices

    def get_count(self):
        return len(self._devices)

    def get_device_by_index(self, index):
        return self._devices[index]


class FakeContext:
    def __init__(self, device_list):
        self._device_list = device_list

    def query_devices(self):
        return self._device_list


class FakePipeline:
    def __init__(self, frame_set):
        self._frame_set = frame_set
        self.started = False

    def start(self, _config):
        self.started = True

    def stop(self):
        self.started = False

    def wait_for_frames(self, _timeout_ms):
        return self._frame_set


class FakeConfig:
    def __init__(self):
        self.streams = []

    def enable_stream(self, profile):
        self.streams.append(profile)


class FakeAlignFilter:
    def __init__(self, **_kwargs):
        pass

    def process(self, frames):
        return frames


class FakeOrbbecBackend:
    """Builds the fake `pyorbbecsdk` objects the camera talks to."""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS, device_count=1, with_depth=True):
        self.color_profiles = FakeStreamProfileList(
            [FakeVideoStreamProfile(width, height, fps, COLOR_FORMAT)]
        )
        depth_profiles = [FakeVideoStreamProfile(width, height, fps, DEPTH_FORMAT)] if with_depth else []
        self.depth_profiles = FakeStreamProfileList(depth_profiles)

        self.color_frame = FakeFrame(width, height, COLOR_FORMAT, fill=64)
        self.depth_frame = FakeFrame(width, height, DEPTH_FORMAT, fill=DEPTH_FILL_MM)
        self.frames = FakeFrameSet(self.color_frame, self.depth_frame)

        infos = [
            FakeDeviceInfo(serial=SERIAL_NUMBER if i == 0 else f"SN{i}", uid=f"uid{i}")
            for i in range(device_count)
        ]
        self.devices = [FakeDevice(info, self.color_profiles, self.depth_profiles) for info in infos]
        self.context = FakeContext(FakeDeviceList(self.devices))
        self.pipeline = None

    def make_context(self):
        return self.context

    def make_pipeline(self, _device=None):
        self.pipeline = FakePipeline(self.frames)
        return self.pipeline

    def make_config(self):
        return FakeConfig()

    def make_align_filter(self, *args, **kwargs):
        return FakeAlignFilter(*args, **kwargs)


@pytest.fixture(autouse=True)
def patch_orbbec_sdk():
    """Replace the Orbbec SDK entry points with the fake backend for every test."""
    backend = FakeOrbbecBackend()
    with (
        patch.object(ob, "Context", side_effect=backend.make_context),
        patch.object(ob, "Pipeline", side_effect=backend.make_pipeline),
        patch.object(ob, "Config", side_effect=backend.make_config),
        patch.object(ob, "AlignFilter", side_effect=backend.make_align_filter),
    ):
        yield backend


# ---------------------------------------------------------------- configuration


def test_is_registered_as_a_camera_config():
    assert OrbbecCameraConfig().type == "orbbec"


def test_defaults_enable_color_and_depth():
    config = OrbbecCameraConfig()
    assert config.use_rgb is True
    assert config.use_depth is True
    assert config.color_mode == ColorMode.RGB
    assert config.rotation == Cv2Rotation.NO_ROTATION
    assert config.serial_number_or_name is None
    assert config.align_depth_to_color is True


def test_requires_at_least_one_stream():
    with pytest.raises(ValueError, match="At least one of"):
        OrbbecCameraConfig(use_rgb=False, use_depth=False)


def test_alignment_requires_both_streams():
    with pytest.raises(ValueError, match="align_depth_to_color"):
        OrbbecCameraConfig(use_rgb=False, align_depth_to_color=True)

    # Dropping the color stream is fine as long as alignment is off too.
    assert OrbbecCameraConfig(use_rgb=False, use_depth=True, align_depth_to_color=False).use_rgb is False
    # Turning alignment off for a color-only camera is also fine.
    assert OrbbecCameraConfig(use_depth=False, align_depth_to_color=False).use_depth is False


def test_resolution_triplet_must_be_complete():
    with pytest.raises(ValueError, match="all of them need to be set"):
        OrbbecCameraConfig(fps=FPS, width=WIDTH)

    assert OrbbecCameraConfig(fps=FPS, width=WIDTH, height=HEIGHT).height == HEIGHT


# ---------------------------------------------------------------- connection


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    camera = OrbbecCamera(OrbbecCameraConfig())
    assert camera.is_connected is False


def test_connect():
    config = OrbbecCameraConfig(warmup_s=0)

    with OrbbecCamera(config) as camera:
        assert camera.is_connected


def test_connect_already_connected():
    config = OrbbecCameraConfig(warmup_s=0)

    with OrbbecCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_connect_without_warmup():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    assert camera.is_connected
    assert camera.serial_number == SERIAL_NUMBER

    camera.disconnect()


def test_connect_unknown_serial_raises():
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="does-not-exist"))

    with pytest.raises(ValueError, match="does-not-exist"):
        camera.connect(warmup=False)


def test_connect_without_any_device_raises():
    empty = FakeOrbbecBackend(device_count=0)

    with (
        patch.object(ob, "Context", side_effect=empty.make_context),
        patch.object(ob, "Pipeline", side_effect=empty.make_pipeline),
    ):
        camera = OrbbecCamera(OrbbecCameraConfig())
        with pytest.raises(ConnectionError, match="no Orbbec camera detected"):
            camera.connect(warmup=False)


@pytest.fixture
def fast_warmup():
    """Shrink the minimum warmup so the failure paths do not really sleep for a second."""
    with patch("lerobot.cameras.orbbec.camera_orbbec._MIN_WARMUP_S", 0.05):
        yield


def test_connect_waits_for_the_streams():
    """`connect()` with warmup on returns once both enabled streams have produced a frame."""
    with OrbbecCamera(OrbbecCameraConfig(warmup_s=0)) as camera:
        assert camera.is_connected
        assert camera.read().shape == (HEIGHT, WIDTH, 3)
        assert camera.read_depth().shape == (HEIGHT, WIDTH, 1)


def test_unsupported_requested_mode_falls_back_to_the_actual_profile(fast_warmup):
    """A sensor may not honour the requested mode. The capture size must then follow what it really
    streams, otherwise every frame is rejected by `_postprocess_image`."""
    config = OrbbecCameraConfig(fps=FPS, width=1280, height=720, warmup_s=0)

    with OrbbecCamera(config) as camera:
        assert (camera.width, camera.height) == (WIDTH, HEIGHT)
        assert (camera.capture_width, camera.capture_height) == (WIDTH, HEIGHT)
        assert camera.read().shape == (HEIGHT, WIDTH, 3)


def test_warmup_reports_the_stream_that_never_produced_a_frame(patch_orbbec_sdk, fast_warmup):
    """A starved depth stream must fail with a message that names it, not a generic timeout."""
    backend = patch_orbbec_sdk
    backend.frames = FakeFrameSet(backend.color_frame, None)

    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))

    with pytest.raises(ConnectionError, match="no depth frame"):
        camera.connect()

    assert backend.pipeline.started is False, "a failed warmup must release the pipeline"


def test_warmup_tolerates_depth_arriving_before_the_color_stream(patch_orbbec_sdk, fast_warmup):
    """The two streams come up independently and share one frame event: the depth sensor reaches
    `STREAMING` before the color one, so the first frame sets is depth-only. That must not abort the
    warmup - doing so made `connect()` fail and the CLI silently fall back to a color-only camera."""
    backend = patch_orbbec_sdk

    class DepthFirstPipeline(FakePipeline):
        def __init__(self):
            super().__init__(None)
            self._calls = 0

        def wait_for_frames(self, _timeout_ms):
            self._calls += 1
            if self._calls == 1:
                return FakeFrameSet(None, backend.depth_frame)
            return FakeFrameSet(backend.color_frame, backend.depth_frame)

    with (
        patch.object(ob, "Pipeline", side_effect=lambda _device=None: DepthFirstPipeline()),
        OrbbecCamera(OrbbecCameraConfig(warmup_s=0)) as camera,
    ):
        assert camera.is_connected
        assert camera.read().shape == (HEIGHT, WIDTH, 3)
        assert camera.read_depth().shape == (HEIGHT, WIDTH, 1)


def test_connect_reports_a_pipeline_that_fails_to_start(patch_orbbec_sdk):
    class FailingPipeline(FakePipeline):
        def start(self, _config):
            raise RuntimeError("UVC interface busy")

    with patch.object(ob, "Pipeline", side_effect=lambda _device=None: FailingPipeline(None)):
        camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
        with pytest.raises(ConnectionError, match="failed to start the Orbbec pipeline"):
            camera.connect(warmup=False)

    assert camera.is_connected is False


def test_connect_reports_a_device_without_the_requested_sensor():
    """`use_depth=True` on a device that only has a color sensor must say so, not fail obscurely."""
    device = FakeDevice(FakeDeviceInfo(), FakeStreamProfileList([FakeVideoStreamProfile()]), None)
    context = FakeContext(FakeDeviceList([device]))

    with patch.object(ob, "Context", side_effect=lambda: context):
        camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
        with pytest.raises(RuntimeError, match="no depth stream profile"):
            camera.connect(warmup=False)


def test_read_still_works_when_only_color_is_available(patch_orbbec_sdk):
    """The software D2C filter only returns a frame set when both streams are present, so an
    incomplete set must bypass it instead of stalling the whole read loop."""
    backend = patch_orbbec_sdk
    backend.frames = FakeFrameSet(backend.color_frame, None)

    class StrictAlignFilter:
        def __init__(self, **_kwargs):
            pass

        def process(self, frames):
            if frames.get_color_frame() is None or frames.get_depth_frame() is None:
                return None
            return frames

    with patch.object(ob, "AlignFilter", side_effect=StrictAlignFilter):
        camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
        camera.connect(warmup=False)
        try:
            assert camera.align_filter is not None
            assert camera.read().shape == (HEIGHT, WIDTH, 3)
            assert camera.read().shape == (HEIGHT, WIDTH, 3)
        finally:
            camera.disconnect()


# ---------------------------------------------------------------- reading


def test_read():
    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
    camera.connect(warmup=False)

    img = camera.read()

    assert isinstance(img, np.ndarray)
    assert img.shape == (HEIGHT, WIDTH, 3)
    assert img.dtype == np.uint8

    camera.disconnect()


def test_read_depth():
    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
    camera.connect(warmup=False)

    depth = camera.read_depth()

    assert isinstance(depth, np.ndarray)
    assert depth.shape == (HEIGHT, WIDTH, 1)
    assert depth.dtype == np.uint16

    camera.disconnect()


def test_color_mode_conversion():
    """RGB and BGR reads of the same frame must differ only by a channel-axis reversal."""
    frames = {}
    for color_mode in (ColorMode.RGB, ColorMode.BGR):
        camera = OrbbecCamera(OrbbecCameraConfig(color_mode=color_mode))
        camera.connect(warmup=False)
        frames[color_mode] = camera.read()
        camera.disconnect()

    assert frames[ColorMode.RGB].shape == frames[ColorMode.BGR].shape
    np.testing.assert_array_equal(frames[ColorMode.RGB], frames[ColorMode.BGR][..., ::-1])


def test_depth_frame_not_color_converted():
    """Depth frames must bypass color conversion, even when a BGR color_mode is set."""
    camera = OrbbecCamera(OrbbecCameraConfig(color_mode=ColorMode.BGR))
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    camera.capture_height, camera.capture_width = depth.shape
    camera.depth_capture_height, camera.depth_capture_width = depth.shape

    np.testing.assert_array_equal(camera._postprocess_image(depth, depth_frame=True), depth)


# ---------------------------------------------------------------- stream formats


def test_format_name_unwraps_the_sdk_enum():
    """`str(ob.OBFormat.Y12)` is "OBFormat.Y12", not "Y12".

    Comparing that against the bare names in `_DEPTH_16BIT_FORMATS`/`_PREFERRED_COLOR_FORMATS`
    matched nothing, so every frame - color and depth alike - was rejected and
    `lerobot-find-cameras orbbec` finished with an empty output directory.
    """
    assert _format_name(ob.OBFormat.Y12) == "Y12"
    assert _format_name(ob.OBFormat.MJPG) == "MJPG"
    # Already-normalized input must survive, so the helper is safe to apply unconditionally.
    assert _format_name("RGB") == "RGB"


@pytest.mark.parametrize(
    "depth_format",
    [ob.OBFormat.Y16, ob.OBFormat.Y12, ob.OBFormat.Y11],
    ids=["y16", "y12", "y11"],
)
def test_depth_formats_reported_by_the_device_are_accepted(depth_format, patch_orbbec_sdk):
    """Y11/Y12 are what an Astra Pro Plus actually streams; Y16 covers the other devices."""
    backend = patch_orbbec_sdk
    backend.frames = FakeFrameSet(
        backend.color_frame, FakeFrame(WIDTH, HEIGHT, depth_format, fill=DEPTH_FILL_MM)
    )

    with OrbbecCamera(OrbbecCameraConfig(warmup_s=0)) as camera:
        depth = camera.read_depth()

    assert depth.shape == (HEIGHT, WIDTH, 1)
    assert depth.dtype == np.uint16
    assert depth.max() == DEPTH_FILL_MM


def test_depth_scale_is_applied_when_the_device_reports_one():
    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
    frame = FakeFrame(WIDTH, HEIGHT, ob.OBFormat.Y16, depth_scale=0.5, fill=2000)

    depth = camera._depth_to_array(frame)

    assert depth.max() == 1000
    assert camera.depth_scale == 0.5


def test_depth_scale_of_zero_does_not_blank_the_map():
    """A device reporting no scale already delivers millimetres; multiplying by 0.0 would turn
    the whole depth map into zeros without any error."""
    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
    frame = FakeFrame(WIDTH, HEIGHT, ob.OBFormat.Y16, depth_scale=0.0, fill=DEPTH_FILL_MM)

    depth = camera._depth_to_array(frame)

    assert depth.max() == DEPTH_FILL_MM


def test_unsupported_color_format_names_the_bare_format():
    camera = OrbbecCamera(OrbbecCameraConfig(warmup_s=0))
    frame = FakeFrame(WIDTH, HEIGHT, ob.OBFormat.H264, fill=0)

    with pytest.raises(ValueError, match="unsupported Orbbec color format 'H264'"):
        camera._color_to_array(frame)


def test_read_before_connect():
    camera = OrbbecCamera(OrbbecCameraConfig())

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_read_color_disabled_raises():
    camera = OrbbecCamera(OrbbecCameraConfig(use_rgb=False, align_depth_to_color=False))
    camera.connect(warmup=False)

    with pytest.raises(RuntimeError, match="use_rgb=False"):
        _ = camera.read()

    camera.disconnect()


def test_read_depth_disabled_raises():
    camera = OrbbecCamera(OrbbecCameraConfig(use_depth=False, align_depth_to_color=False))
    camera.connect(warmup=False)

    with pytest.raises(RuntimeError, match="use_depth=False"):
        _ = camera.read_depth()

    camera.disconnect()


# ---------------------------------------------------------------- disconnect / async


def test_disconnect():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    camera = OrbbecCamera(OrbbecCameraConfig())

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    img = camera.async_read()

    assert camera.thread is not None
    assert camera.thread.is_alive()
    assert isinstance(img, np.ndarray)

    camera.disconnect()


def test_async_read_depth():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    depth = camera.async_read_depth()

    assert isinstance(depth, np.ndarray)
    assert depth.dtype == np.uint16

    camera.disconnect()


def test_async_read_before_connect():
    camera = OrbbecCamera(OrbbecCameraConfig())

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


# ---------------------------------------------------------------- read_latest


def test_read_latest():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    frame = camera.read()
    latest = camera.read_latest()

    assert isinstance(latest, np.ndarray)
    assert latest.shape == frame.shape

    camera.disconnect()


def test_read_latest_depth():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    frame = camera.read_depth()
    latest = camera.read_latest_depth()

    assert isinstance(latest, np.ndarray)
    assert latest.shape == frame.shape
    assert latest.dtype == np.uint16

    camera.disconnect()


def test_read_latest_before_connect():
    camera = OrbbecCamera(OrbbecCameraConfig())

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_high_frequency():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    # prime to ensure frames are available
    ref = camera.read()

    for _ in range(20):
        latest = camera.read_latest()
        assert isinstance(latest, np.ndarray)
        assert latest.shape == ref.shape

    camera.disconnect()


def test_read_latest_too_old():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera.connect(warmup=False)

    # prime to ensure frames are available
    _ = camera.read()

    with pytest.raises(TimeoutError):
        _ = camera.read_latest(max_age_ms=0)  # immediately too old

    camera.disconnect()


# ---------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
def test_rotation(rotation):
    camera = OrbbecCamera(OrbbecCameraConfig(rotation=rotation, warmup_s=0))
    camera.connect(warmup=False)

    img = camera.read()

    assert isinstance(img, np.ndarray)
    if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
        assert camera.width == HEIGHT
        assert camera.height == WIDTH
        assert img.shape[:2] == (WIDTH, HEIGHT)
    else:
        assert camera.width == WIDTH
        assert camera.height == HEIGHT
        assert img.shape[:2] == (HEIGHT, WIDTH)

    camera.disconnect()


# ---------------------------------------------------------------- discovery


def test_find_cameras():
    cameras = OrbbecCamera.find_cameras()

    assert len(cameras) == 1
    info = cameras[0]
    assert info["type"] == "Orbbec"
    assert info["id"] == SERIAL_NUMBER
    assert info["name"] == DEVICE_NAME
    # Bare names, not the raw `str()` of the enum ("OBFormat.RGB").
    assert info["default_color_stream_profile"]["format"] == "RGB"
    assert info["default_depth_stream_profile"]["format"] == "Y16"


def test_find_cameras_falls_back_to_uid_when_serial_number_is_empty():
    """An Astra Pro Plus on a USB 2.0 link reports an empty serial number."""
    info = FakeDeviceInfo(serial="", name=DEVICE_NAME, uid="uid-42")
    # Color has a sensor with no profiles, depth has no sensor at all: both must be skipped
    # silently rather than aborting the enumeration.
    device = FakeDevice(info, FakeStreamProfileList([]), None)
    context = FakeContext(FakeDeviceList([device]))

    with patch.object(ob, "Context", side_effect=lambda: context):
        cameras = OrbbecCamera.find_cameras()

    assert cameras[0]["id"] == "uid-42"
    assert cameras[0]["serial_number"] == ""
    assert "default_color_stream_profile" not in cameras[0]
    assert "default_depth_stream_profile" not in cameras[0]


def test_find_cameras_propagates_unrelated_sdk_errors():
    class BrokenContext:
        def query_devices(self):
            raise RuntimeError("some unrelated SDK failure")

    with (
        patch.object(ob, "Context", side_effect=BrokenContext),
        pytest.raises(RuntimeError, match="some unrelated SDK failure"),
    ):
        OrbbecCamera.find_cameras()


def test_empty_serial_number_matches_a_single_device():
    """`serial_number_or_name=""` is what a device without a serial number produces; it must not
    be treated as a filter that matches nothing."""
    with OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="", warmup_s=0)) as camera:
        assert camera.is_connected


def test_cli_color_only_fallback_connects(patch_orbbec_sdk, fast_warmup):
    """`lerobot-find-cameras orbbec` retries a camera with `use_depth=False` when depth never
    arrives. That fallback used to be dead code: the resulting config was rejected outright, so
    the command silently produced no camera - and therefore no images - at all.
    """
    backend = patch_orbbec_sdk
    backend.frames = FakeFrameSet(backend.color_frame, None)  # depth never delivers

    result = create_camera_instance({"type": "Orbbec", "id": SERIAL_NUMBER}, warmup_s=0)

    try:
        assert result is not None, "the color-only fallback must still yield a usable camera"
        assert result["instance"].use_depth is False
        assert result["instance"].is_connected
    finally:
        if result is not None:
            result["instance"].disconnect()
