# setup_motor — SO-ARM101 舵机配置（教学精简版）

这是 `lerobot-setup-motors` 的解耦复刻：不依赖 lerobot 库，只依赖 Feetech 官方 SDK，
几百行代码把"给 6 只舵机分配 ID、统一波特率"这件事讲清楚。适合对照 lerobot 源码阅读。

## 文件结构与阅读顺序

| 文件 | 内容 |
|---|---|
| `tables.py` | STS3215 的寄存器地址、波特率编码表（"协议手册"），先读这个 |
| `sts3215_bus.py` | 极简总线驱动：开串口、广播 PING、读写寄存器 |
| `setup_motors.py` | 主脚本：交互式逐个设置 6 只舵机（对应 `lerobot-setup-motors`） |
| `check_motors.py` | 只读体检：验证总线状态与 lerobot 预期一致、展示/比对标定值 |
| `verify_with_lerobot.py` | 用 lerobot 库自己的驱动+握手验证 setup 结果（检查路径 100% 是 lerobot 代码） |

## 运行

```bash
conda activate so101
cd setup_motor
python setup_motors.py               # 自动探测串口
python setup_motors.py /dev/ttyACM0  # 或手动指定
```

依赖：`feetech-servo-sdk`、`pyserial`（so101 环境已装好）。

## 操作步骤

1. 电源（7.4V/12V 按舵机版本）和 USB 都接好——**USB 不给舵机供电**。
2. 按提示从 **gripper（6 号）** 开始，每次只用一根三针线把当前舵机单独接到驱动板，
   它不得与任何其他舵机相连，按回车，看到"完成"后换下一只。
3. 顺序：gripper → wrist_roll → wrist_flex → elbow_flex → shoulder_lift → shoulder_pan。
   全部完成后把舵机按链串回，1 号（shoulder_pan）接驱动板。

设置过的舵机不需要重新设置（ID/波特率写在 EEPROM，掉电保存）；
中途 Ctrl+C 退出是安全的。

## 每只舵机背后发生了什么

对应 `setup_motors.py: setup_one_motor` 的五步：

1. **探测**（`find_single_motor`）：从 1M 到 19.2K 逐档切串口波特率，每档广播一次 PING，
   应答的那只就是目标，顺便校验型号必须是 sts3215（型号码 777）。
2. **解锁 EEPROM**：写 `Torque_Enable=0`（地址 40）、`Lock=0`（地址 55）——
   不松力矩舵机拒绝修改 EEPROM。
3. **写 ID**（地址 5）：用旧 ID 寻址写新 ID，生效后舵机立即"改名"。
4. **写波特率**（地址 6）：写编码值（1M 对应 0，不是 1000000）。
5. **验证**：串口切到 1M，用新 ID 读型号，读得通才算成功。

## 与 lerobot 源码的对应关系

| 本项目 | lerobot 源码 |
|---|---|
| `setup_motors.py` 交互循环 | `src/lerobot/robots/so_follower/so_follower.py` 的 `setup_motors()` |
| `find_single_motor` | `src/lerobot/motors/feetech/feetech.py` 的 `_find_single_motor_p0` |
| `sts3215_bus.broadcast_ping` | 同文件 `_broadcast_ping`（原始包解析的精简版） |
| `setup_one_motor` 的写 ID/波特率 | `src/lerobot/motors/motors_bus.py` 的 `setup_motor()` |
| `tables.py` | `src/lerobot/motors/feetech/tables.py` |

与 lerobot 的一个有意差异：lerobot 探测到多只舵机应答时只取第一个、不报错，
误接整条链会把不该动的舵机 ID 改乱；本实现检测到多个应答直接报错，对初学者更安全。

## 常见报错

| 报错 | 原因 |
|---|---|
| `总线上没有发现舵机` | 三针线没插好/插反；电源没上电；舵机没接在驱动板上 |
| `总线上有 N 只舵机应答` | 链没断开，还有别的舵机挂在总线上，断开再按回车 |
| `探测到的舵机型号是 ... 不是 sts3215` | 接错了舵机（或总线上挂着非 sts3215 设备） |
| `无法打开串口` | 端口名不对、被其他程序占用，或没有权限（可 `sudo` 或把用户加入 `dialout` 组） |
