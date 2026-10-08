"""lerobot-setup-motors 的教学精简版。

功能：给 SO-ARM101 的每只 sts3215 舵机写入唯一 ID，并把波特率统一为 1M。
这些参数写入舵机 EEPROM，掉电保存，每只舵机只需设置一次。

用法：
    conda activate so101
    python setup_motors.py               # 自动探测串口
    python setup_motors.py /dev/ttyACM0  # 手动指定串口

硬件要求：每按一次回车前，驱动板上必须只连接了提示的那一只舵机，
原因见 find_single_motor 的注释。
"""

import argparse
import glob
import sys

from sts3215_bus import STS3215Bus
from tables import (
    ADDR_BAUD_RATE,
    ADDR_ID,
    BAUDRATE_TABLE,
    DEFAULT_BAUDRATE,
    SCAN_BAUDRATES,
    STS3215_MODEL_NUMBER,
)

# 六个关节舵机的目标 ID，与 lerobot 的定义保持一致
# （src/lerobot/robots/so_follower/so_follower.py 中的 self.bus = FeetechMotorsBus(...)）
MOTORS = {
    "shoulder_pan": 1,  # 底座旋转
    "shoulder_lift": 2,  # 肩部抬放
    "elbow_flex": 3,  # 肘部
    "wrist_flex": 4,  # 腕部俯仰
    "wrist_roll": 5,  # 腕部旋转（可整圈）
    "gripper": 6,  # 夹爪
}


def find_single_motor(bus: STS3215Bus) -> tuple[int, int]:
    """找出当前接在总线上那一只舵机，返回 (它当前的波特率, 它当前的 ID)。

    程序并不知道舵机出厂的 ID 和波特率，只能暴力扫描：把串口切到每个
    波特率试一遍，广播 PING，谁应答谁就是它。

    "只接一只舵机"是硬性要求：如果两只同时应答，程序无法区分谁是目标，
    这里直接报错。（lerobot 的 _find_single_motor_p0 只取第一个应答并不
    校验数量，整条链没断开时会静默改错舵机的 ID，这里做了加固。）
    """
    for baudrate in SCAN_BAUDRATES:
        bus.set_baudrate(baudrate)
        ids = bus.broadcast_ping()
        if not ids:
            continue
        if len(ids) > 1:
            raise RuntimeError(
                f"总线上有 {len(ids)} 只舵机应答（ID: {ids}）。"
                "请只连接当前提示的那一只舵机！"
            )
        model = bus.read_model_number(ids[0])
        if model != STS3215_MODEL_NUMBER:
            raise RuntimeError(
                f"探测到的舵机型号是 {model}，不是预期的 sts3215"
                f"（{STS3215_MODEL_NUMBER}）。检查是否接错了舵机。"
            )
        return baudrate, ids[0]
    raise RuntimeError(
        "总线上没有发现舵机。检查：三针线是否插好、电源是否上电（USB 不给舵机供电）。"
    )


def setup_one_motor(bus: STS3215Bus, name: str, target_id: int) -> None:
    """对一只舵机完成：探测 → 解锁 EEPROM → 写 ID → 写波特率 → 验证。"""
    initial_baudrate, initial_id = find_single_motor(bus)
    bus.set_baudrate(initial_baudrate)  # 此后所有通信都在舵机当前的波特率上进行
    bus.disable_torque(initial_id)

    # 用旧 ID 寻址写入新 ID；这条指令生效后，舵机立即以新 ID 应答
    bus.write_byte(initial_id, ADDR_ID, target_id)

    # 波特率寄存器存的是编码值（1M 对应 0，见 tables.BAUDRATE_TABLE）。
    # 注意寻址用的是新 ID——上一步之后舵机已经"改名"了
    bus.write_byte(target_id, ADDR_BAUD_RATE, BAUDRATE_TABLE[DEFAULT_BAUDRATE])

    # 串口也切到 1M，再用新 ID 读一次型号：能读通说明 ID 和波特率都改对了
    bus.set_baudrate(DEFAULT_BAUDRATE)
    model = bus.read_model_number(target_id)
    if model != STS3215_MODEL_NUMBER:
        raise RuntimeError(f"验证失败：新 ID {target_id} 上读到的型号是 {model}")


def pick_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise SystemExit("未发现串口设备（/dev/ttyUSB*、/dev/ttyACM*）。插好驱动板，或手动指定端口。")
    print("发现多个串口，请选择：")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    return ports[int(input("输入编号: ").strip())]


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-ARM101 舵机 ID/波特率设置工具")
    parser.add_argument("port", nargs="?", help="串口路径，如 /dev/ttyACM0（不填则自动探测）")
    args = parser.parse_args()

    port = pick_port(args.port)
    bus = STS3215Bus(port)

    opened = False
    try:
        bus.open()
        opened = True
        print(f"串口已打开: {port}\n")
        print("电源（匹配电压）和 USB 都要接好；USB 不给舵机供电。")

        for name in reversed(MOTORS):  # 从 gripper(6) 开始，到底座 shoulder_pan(1) 结束
            target_id = MOTORS[name]
            input(f"\n>>> 只把驱动板接到 '{name}' 舵机（将设为 ID {target_id}），按回车继续...")
            setup_one_motor(bus, name, target_id)
            print(f"    完成：'{name}' ID = {target_id}，波特率 = {DEFAULT_BAUDRATE}")
    except KeyboardInterrupt:
        print("\n\n已中断。已设置过的舵机无需重设（参数存在 EEPROM，掉电保存）。")
        return
    except (ConnectionError, RuntimeError, OSError) as e:
        print(f"\n出错：{e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n6 只舵机全部设置完成！现在把它们按链条串回去（1 号 shoulder_pan 接驱动板）。")
    finally:
        if opened:
            bus.close()


if __name__ == "__main__":
    main()
