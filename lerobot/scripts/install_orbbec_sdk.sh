#!/usr/bin/env bash
#
# Build and install `pyorbbecsdk` so that LeRobot's OrbbecCamera backend works.
#
# Why a script instead of a `lerobot[orbbec]` extra: Orbbec does not publish `pyorbbecsdk` on
# PyPI, and its two SDK generations support disjoint device sets:
#   * branch `main`    -> Orbbec SDK v1.x, required by older devices
#                         (Astra, Astra+, Astra Pro Plus, Astra Mini, Dabai, ...)
#   * branch `v2-main` -> Orbbec SDK v2.x, required by newer devices
#                         (Gemini 330/335/336, Gemini 335Le, Gemini 215/210, ...)
# This script inspects the USB IDs of the connected camera and picks the matching branch.
#
# Usage:
#   scripts/install_orbbec_sdk.sh                 # auto-detect the branch
#   scripts/install_orbbec_sdk.sh --branch v1     # force the v1 SDK (`main`)
#   scripts/install_orbbec_sdk.sh --branch v2     # force the v2 SDK (`v2-main`)
#   JOBS=2 scripts/install_orbbec_sdk.sh          # limit compiler parallelism (default: 4)
#
# Linux only: on Linux the SDK talks to the camera through libusb, so the udev rules must be
# installed once (the script prints the exact command).

set -euo pipefail

ORBBEC_SRC="${ORBBEC_SRC:-$HOME/pyorbbecsdk}"
REPO_URL="https://github.com/orbbec/pyorbbecsdk.git"
JOBS="${JOBS:-4}"
BRANCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      case "${2:-}" in
        v1 | main) BRANCH="main" ;;
        v2 | v2-main) BRANCH="v2-main" ;;
        *)
          echo "error: --branch expects 'v1'/'main' or 'v2'/'v2-main', got '${2:-}'" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --src)
      ORBBEC_SRC="${2:?--src needs a path}"
      shift 2
      ;;
    -h | --help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ── 1. Detect the connected device and choose the SDK generation ─────────────────────────────
# Product IDs that are only serviced by the legacy SDK v1 (OpenNI-era sensors). Anything else is
# assumed to be a newer sensor, for which Orbbec recommends SDK v2.
V1_ONLY_PIDS="0401 0402 0403 0404 0407 0501 050b 050e 050f 0510 0511 0516 0517 0518 051b 0601 0603 060b 060e 060f 0610 0614 0616 0617 0618 061b 062b 0559 0655 0656 0659 065a 069a 069e"

if [[ -z "$BRANCH" ]]; then
  log "Detecting connected Orbbec device"
  detected_pids=""
  for dev in /sys/bus/usb/devices/*/idVendor; do
    [[ -r "$dev" ]] || continue
    [[ "$(cat "$dev")" == "2bc5" ]] || continue
    pid_file="${dev%idVendor}idProduct"
    [[ -r "$pid_file" ]] || continue
    detected_pids+=" $(cat "$pid_file")"
  done

  if [[ -z "${detected_pids// /}" ]]; then
    echo "warning: no Orbbec device (USB vendor 2bc5) found on this machine; defaulting to SDK v2." >&2
    BRANCH="v2-main"
  else
    echo "  Orbbec product IDs found:$detected_pids"
    BRANCH="v2-main"
    for pid in $detected_pids; do
      for legacy in $V1_ONLY_PIDS; do
        if [[ "$pid" == "$legacy" ]]; then
          BRANCH="main"
          break 2
        fi
      done
    done
  fi
fi

if [[ "$BRANCH" == "main" ]]; then
  echo "  -> using branch 'main' (Orbbec SDK v1.x)"
else
  echo "  -> using branch 'v2-main' (Orbbec SDK v2.x)"
fi

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: '$python_bin' not found. Activate the environment where lerobot lives." >&2
  exit 1
fi
echo "  interpreter: $("$python_bin" -c 'import sys; print(sys.executable)')"

# ── 2. Fetch the sources ────────────────────────────────────────────────────────────────────
if [[ -d "$ORBBEC_SRC/.git" ]]; then
  log "Reusing existing checkout at $ORBBEC_SRC"
  git -C "$ORBBEC_SRC" fetch --depth 1 origin "$BRANCH"
  git -C "$ORBBEC_SRC" checkout -q FETCH_HEAD
else
  log "Cloning pyorbbecsdk ($BRANCH) into $ORBBEC_SRC"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ORBBEC_SRC"
fi

# ── 3. Build ────────────────────────────────────────────────────────────────────────────────
# pybind11 >= 2.12 is required for NumPy 2.x: the SDK hands frames back as `py::array_t`, which
# pybind11 2.11 cannot convert against NumPy 2. LeRobot pins NumPy >= 2, so install a new one.
log "Installing build dependencies"
"$python_bin" -m pip install -q "pybind11>=2.13,<3"

log "Configuring (branch $BRANCH)"
mkdir -p "$ORBBEC_SRC/build"
pushd "$ORBBEC_SRC/build" >/dev/null
cmake -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$("$python_bin" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DBUILD_TESTING=OFF ..

log "Compiling with -j$JOBS"
cmake --build . -- -j"$JOBS"
cmake --install .
popd >/dev/null

# ── 4. Package and install ──────────────────────────────────────────────────────────────────
log "Building the wheel"
pushd "$ORBBEC_SRC" >/dev/null
"$python_bin" -m pip install -q wheel
"$python_bin" setup.py -q bdist_wheel
wheel="$(ls -t dist/*.whl | head -1)"
echo "  $wheel"
"$python_bin" -m pip install --no-deps --force-reinstall "$wheel"
popd >/dev/null

# ── 5. Verify ───────────────────────────────────────────────────────────────────────────────
log "Verifying the installation"
if ! "$python_bin" - <<'PY'
import sys

try:
    import pyorbbecsdk as ob
except Exception as e:  # noqa: BLE001
    print(f"  import failed: {e}")
    sys.exit(1)

print(f"  pyorbbecsdk imported from {ob.__file__}")
try:
    count = ob.Context().query_devices().get_count()
except Exception as e:  # noqa: BLE001
    print(f"  device enumeration failed: {e}")
    print("  (on Linux this usually means the udev rules are missing - see below)")
    sys.exit(0)

print(f"  {count} Orbbec device(s) detected")
if count == 0:
    print("  If your camera is plugged in, try the other SDK branch:")
    print("    scripts/install_orbbec_sdk.sh --branch v1   # older Astra / Dabai / Astra Pro Plus")
    print("    scripts/install_orbbec_sdk.sh --branch v2   # newer Gemini 330/335/336")
    sys.exit(0)

for i in range(count):
    info = ob.Context().query_devices().get_device_by_index(i).get_device_info()
    print(f"    [{i}] {info.get_name()}  SN={info.get_serial_number()}  PID={hex(info.get_pid())}")
PY
then
  echo "error: verification failed" >&2
  exit 1
fi

RULES_FILE="$(find "$ORBBEC_SRC" -name '99-obsensor-libusb.rules' -print -quit 2>/dev/null || true)"

cat <<EOF

Done. Next steps:

  1. (Linux, once) install the udev rules so the SDK can open the USB device without root:
       sudo cp "${RULES_FILE:-$ORBBEC_SRC/99-obsensor-libusb.rules}" /etc/udev/rules.d/
       sudo udevadm control --reload-rules && sudo udevadm trigger
       # then unplug and replug the camera

  2. List the camera and dump a test color + depth frame:
       lerobot-find-cameras orbbec

  3. Use it from Python:
       from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
       with OrbbecCamera(OrbbecCameraConfig()) as cam:
           color = cam.read()        # (H, W, 3) uint8
           depth = cam.read_depth()  # (H, W, 1) uint16, millimetres
EOF
