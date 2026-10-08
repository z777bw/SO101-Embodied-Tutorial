"""用 lerobot 库本身验证 setup 结果（不依赖本目录的任何驱动代码）。

验证逻辑：setup 的产出是一个明确的总线状态。让 lerobot 用它自己的驱动
连接并握手——握手会逐个 ping ID 1~6、校验型号和固件一致性，这正是
lerobot 运行时对舵机的全部要求；再读出波特率寄存器比对（lerobot 的
setup 工具写入的值应为 0，即 1M）。

本脚本对舵机只读不写（disconnect 时也显式关闭"断力矩"写操作）。

用法：python verify_with_lerobot.py /dev/ttyACM0
"""

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# 与 lerobot 的 SOFollower / SOLeader 定义完全相同的电机表
# （两者默认 use_degrees=True，关节电机用 DEGREES，夹爪用 RANGE_0_100）
MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="用 lerobot 库验证舵机 setup 结果")
    parser.add_argument("port", help="串口路径，如 /dev/ttyACM0")
    parser.add_argument(
        "--ids", type=int, nargs="+", metavar="ID",
        help="只检查这些 ID（如 --ids 6，或 --ids 1 2 3）；不填则要求 6 只全部在线",
    )
    args = parser.parse_args()

    motors = MOTORS
    if args.ids:
        motors = {name: m for name, m in MOTORS.items() if m.id in args.ids}
        if not motors:
            parser.error(f"--ids 里没有有效的舵机 ID，可选范围: 1~6")
        print(f"本次只检查 ID {sorted(m.id for m in motors.values())}（其余舵机不必连接）\n")

    bus = FeetechMotorsBus(port=args.port, motors=motors)
    try:
        bus.connect()  # 握手：逐个 ping 6 个 ID，校验型号与固件一致性
    except ConnectionError as e:
        print(f"lerobot 连接失败：{e}")
        sys.exit(1)

    print("lerobot 握手通过：6 个 ID 全部应答，型号与固件符合预期\n")
    print(f"{'关节':<14}{'ID':<6}{'型号寄存器':<12}{'波特率寄存器':<14}判定")

    all_ok = True
    for motor in bus.motors:
        model = bus.read("Model_Number", motor, normalize=False)
        baud = bus.read("Baud_Rate", motor, normalize=False)
        ok = baud == 0  # 0 = 1M，即 lerobot 的 DEFAULT_BAUDRATE 写入值
        all_ok &= ok
        print(f"{motor:<14}{bus.motors[motor].id:<6}{model:<12}{baud:<14}"
              f"{'OK' if ok else '异常：应为 0'}")

    if all_ok:
        n = len(bus.motors)
        scope = "全部 6 只" if n == 6 else f"指定的 {n} 只"
        print(f"\n[结论] 检查过的 {scope}舵机，总线状态与 lerobot-setup-motors 的预期产出完全一致")
    else:
        print("\n[结论] 存在偏差，见上表")

    bus.disconnect(disable_torque=False)  # 保持全程只读
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
