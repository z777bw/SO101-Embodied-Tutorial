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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Configuration class for Orbbec RGB-D cameras.

    This class provides specialized configuration options for Orbbec cameras driven by the
    `pyorbbecsdk` wrapper of the Orbbec SDK, including depth sensing and device identification
    via serial number or device name.

    Example configurations for an Orbbec Astra Pro Plus:
    ```python
    # Let lerobot pick the only connected device, color + depth (defaults)
    OrbbecCameraConfig()

    # Explicit device, VGA RGB only
    OrbbecCameraConfig("0123456789", 30, 640, 480, use_depth=False)

    # Depth only, no color stream, rotated 90°
    OrbbecCameraConfig("0123456789", 30, 640, 480, use_rgb=False, rotation=Cv2Rotation.ROTATE_90)
    ```

    Attributes:
        serial_number_or_name: Unique serial number or human-readable device name used to identify
            the camera. When ``None``, the first detected Orbbec device is used, which is
            convenient when only one camera is plugged in.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_rgb: Whether to enable the color stream. Defaults to True.
        use_depth: Whether to enable the depth stream. Defaults to True.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds).
        align_depth_to_color: Whether to align the depth map onto the color stream (D2C) using the
            software align filter, so that both frames share the same field of view. Only
            meaningful when both streams are enabled. Defaults to True.

    Note:
        - At least one of `use_rgb` or `use_depth` must be enabled.
        - `align_depth_to_color` requires both `use_rgb` and `use_depth`.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
        - Depth frames are always returned as `np.uint16` in millimetres, matching
          `RealSenseCamera`, regardless of the device's internal depth scale.
    """

    serial_number_or_name: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = True
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    align_depth_to_color: bool = True

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        if self.align_depth_to_color and not (self.use_rgb and self.use_depth):
            raise ValueError("`align_depth_to_color` requires both `use_rgb` and `use_depth` to be enabled.")

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
