"""STS3215 舵机的"寄存器手册"。

舵机内部有一张控制表：每个参数占用一个地址，往地址写字节就能修改配置。
这里只列出 setup 用到的几项，完整表见 lerobot 源码
src/lerobot/motors/feetech/tables.py 或 Feetech 官方手册。
"""

# sts3215 的型号编码，存放在地址 3（2 字节，只读），用来确认总线上接的是预期型号
MODEL_NUMBER_ADDR = 3
STS3215_MODEL_NUMBER = 777

# ---- 控制表条目（地址, 字节数）----
ADDR_ID = 5              # 舵机在总线上的唯一编号，1 字节，范围 0~254
ADDR_BAUD_RATE = 6       # 波特率，1 字节
ADDR_MIN_POSITION_LIMIT = 9   # 行程下限，2 字节（calibrate 写入）
ADDR_MAX_POSITION_LIMIT = 11  # 行程上限，2 字节（calibrate 写入）
ADDR_TORQUE_ENABLE = 40  # 力矩开关：0 = 放松（可徒手掰动），1 = 锁轴
ADDR_LOCK = 55           # EEPROM 写保护：0 = 允许写，1 = 只读
ADDR_HOMING_OFFSET = 31  # 中位偏置，2 字节（calibrate 写入）

# 波特率编码表：寄存器里存的是右边的"编码值"，不是真实波特率。
BAUDRATE_TABLE = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

# lerobot 要求 6 只舵机统一工作在这个波特率，sync_read/sync_write 才能高速同步
DEFAULT_BAUDRATE = 1_000_000

# 探测舵机时依次尝试的波特率，从高到低（正常情况下 1M 一档就命中）
SCAN_BAUDRATES = sorted(BAUDRATE_TABLE, reverse=True)
