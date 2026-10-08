"""极简的 STS3215 串行总线驱动，只实现 setup 所需的几个操作。

它是对 Feetech 官方 SDK（scservo_sdk）的薄封装，思路与 lerobot 的
FeetechMotorsBus（src/lerobot/motors/feetech/feetech.py）一致：
组包、校验、串口收发这些字节层面的活交给 SDK，这里只写业务逻辑。
"""

import scservo_sdk as scs

from tables import ADDR_LOCK, ADDR_TORQUE_ENABLE


def patch_set_packet_timeout(self, packet_length: int) -> None:
    """修复 PyPI 版 SDK 的超时计算 bug（低波特率下超时太短容易误报通信失败）。

    与 lerobot 中的补丁相同，见 src/lerobot/motors/feetech/feetech.py 的 patch_setPacketTimeout。
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class STS3215Bus:
    def __init__(self, port: str):
        self.port_handler = scs.PortHandler(port)
        # 协议版本 0 是 SCS/STS 系列舵机（包括 sts3215）使用的协议
        self.packet_handler = scs.PacketHandler(0)
        self.port_handler.setPacketTimeout = patch_set_packet_timeout.__get__(
            self.port_handler, scs.PortHandler
        )

    def open(self) -> None:
        # 设备名不对/没权限时 SDK 底下的 pyserial 会直接抛异常，这里兜住转成友好提示
        try:
            ok = self.port_handler.openPort()
        except OSError as e:
            raise OSError(
                f"无法打开串口 '{self.port_handler.port_name}'，检查设备名、权限或是否被占用\n{e}"
            ) from e
        if not ok:
            raise OSError(f"无法打开串口 '{self.port_handler.port_name}'")
        if not self.port_handler.setBaudRate(1_000_000):
            raise OSError("设置串口波特率失败")

    def close(self) -> None:
        self.port_handler.closePort()

    def set_baudrate(self, baudrate: int) -> None:
        """把电脑串口的波特率切到 baudrate（舵机那头不变，两边对上才通）。"""
        if not self.port_handler.setBaudRate(baudrate):
            raise OSError(f"串口不支持波特率 {baudrate}")

    def broadcast_ping(self) -> list[int]:
        """向 ID 0xFE 广播 PING 指令，返回所有应答舵机的 ID 列表。

        广播包：FF FF [ID=0xFE] [LEN=2] [INSTR=0x01] [CHK]，组包交给 SDK 的 txPacket。
        每只应答的舵机回一个 6 字节状态包：FF FF [ID] [LEN] [ERR] [CHK]。
        实现对应 lerobot FeetechMotorsBus._broadcast_ping 的精简版。
        """
        txpacket = [0] * 6
        txpacket[scs.PKT_ID] = scs.BROADCAST_ID
        txpacket[scs.PKT_LENGTH] = 2
        txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING
        # 广播没有"接收应答"的收尾环节，SDK 的 txPacket 发完不会自动清 is_using，
        # 必须手动复位，否则下一次通信会报 "Port is in use!"（lerobot 同款处理）
        if self.packet_handler.txPacket(self.port_handler, txpacket) != scs.COMM_SUCCESS:
            self.port_handler.is_using = False
            return []

        # 最长可能收到 MAX_ID 个状态包，按字节数估算超时时间
        status_length = 6
        wait_length = status_length * scs.MAX_ID
        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0
        self.port_handler.setPacketTimeoutMillis(
            (wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0
        )

        rxpacket: list[int] = []
        while not self.port_handler.isPacketTimeout() and len(rxpacket) < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - len(rxpacket))
        self.port_handler.is_using = False

        ids = []
        while len(rxpacket) >= status_length:
            if not (rxpacket[0] == 0xFF and rxpacket[1] == 0xFF):
                del rxpacket[0]  # 噪声字节，跳过继续找包头
                continue
            checksum = ~sum(rxpacket[2 : status_length - 1]) & 0xFF
            if rxpacket[status_length - 1] == checksum:
                ids.append(rxpacket[scs.PKT_ID])
                del rxpacket[0:status_length]
            else:
                del rxpacket[0:2]  # 假包头：只丢掉这两个字节重新找（与 lerobot 同款处理）
        return sorted(set(ids))

    def read_model_number(self, servo_id: int) -> int:
        model, comm, err = self.packet_handler.read2ByteTxRx(
            self.port_handler, servo_id, 3  # 3 = 型号码寄存器地址
        )
        self._check(comm, err, f"读取型号（id={servo_id}）")
        return model

    def write_byte(self, servo_id: int, address: int, value: int) -> None:
        comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, servo_id, address, value)
        self._check(comm, err, f"写地址 {address}（id={servo_id}）")

    def disable_torque(self, servo_id: int) -> None:
        # ID、波特率这类参数存在 EEPROM 里，必须先松力矩、解写保护才允许修改
        self.write_byte(servo_id, ADDR_TORQUE_ENABLE, 0)
        self.write_byte(servo_id, ADDR_LOCK, 0)

    def _check(self, comm: int, err: int, what: str) -> None:
        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(f"{what} 通信失败: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"{what} 舵机返回错误: {self.packet_handler.getRxPacketError(err)}")
