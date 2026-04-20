"""
config.py — 全局参数配置
所有路径、滤波参数、绘图设置集中在此，两个步骤脚本均从此导入。
"""
import os

# ── 目录 ──────────────────────────────────────────────────────────────────
BASE         = r'g:\Cursor project\ZCY-shengfashe'          # 项目根目录
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录
RESULT_DIR   = os.path.join(PIPELINE_DIR, '结果')
STEP1_DIR    = os.path.join(RESULT_DIR, 'step1_自动滤波')
STEP2_DIR    = os.path.join(RESULT_DIR, 'step2_手动清理')

# ── 原始数据文件 ───────────────────────────────────────────────────────────
US_FILE  = os.path.join(BASE, '超声波', '04-15 - ultrasonics data.csv')
CAL_FILE = os.path.join(BASE, '超声波', 'chushi.csv')
AE_HITS  = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')
AE_EVTS  = os.path.join(BASE, '声发射', '04-15-声发射事件.TXT')

# Step1 输出（Step2 作为输入读取）
STEP1_AE_CLEAN  = os.path.join(STEP1_DIR, 'AE_clean.csv')
STEP1_AE_CONTAM = os.path.join(STEP1_DIR, 'AE_contaminated.csv')

# Step2 输出
STEP2_AE_CLEAN   = os.path.join(STEP2_DIR, 'AE_final_clean.csv')
STEP2_AE_DELETED = os.path.join(STEP2_DIR, 'AE_manual_deleted.csv')

# ── 试样参数 ───────────────────────────────────────────────────────────────
H_MM = 100.0       # 试样高度 (mm)
H_M  = H_MM / 1000.0
D_MM = 50.0        # 试样直径 (mm)；用于空间分布图轮廓绘制

# ── 超声波处理参数 ─────────────────────────────────────────────────────────
FS_HZ         = 40e6     # 采样频率 (Hz)
BP_LOW_HZ     = 50e3     # 带通低截止 (Hz)
BP_HIGH_HZ    = 700e3    # 带通高截止 (Hz)
BP_ORDER      = 4
AIC_OFFSET_US = 12.0     # AIC 搜索窗口：参考点前偏移 (μs)
AIC_WIDTH_US  = 30.0     # AIC 搜索窗口：参考点后宽度 (μs)
AIC_START_US  = 5.0      # AIC 搜索全局起始 (μs)

# ── 声发射自动滤波参数 ─────────────────────────────────────────────────────
# 时间窗（用于拟合参考 hits 及过滤条件）
US_MASK_PRE  = 0.05    # US 脉冲前窗口 (s)
US_MASK_POST = 1.00    # US 脉冲后窗口 (s)

# 干扰条带半宽 (dB)  — 各通道独立配置，0 = 不过滤
CH_STRIPE_HW = {1: 5, 2: 5, 3: 4, 4: 0, 5: 6, 6: 5}
CH_NO_FILTER = {4}     # 该集合内的通道完全跳过干扰滤波

# 条带中心拟合参数
T_BIN          = 60.0   # 时间箱宽度 (s)
SIGMA_BINS     = 2      # 高斯平滑宽度 (bins)
MIN_REF_N      = 3      # 每箱最少参考 hits 数
FIT_END_MARGIN = 200.0  # 安静期末端排除多少秒（避免爆发前 AE 污染拟合）
EXTRAP_LAST_N  = 5      # 恒值外推：取安静期最后多少个有效箱均值

# 爆发段处理
BURST_DILUTE    = 0.50  # 爆发段条带内随机保留比例（0.5 = 保留 50%）
BURST_RAND_SEED = 42    # 随机种子（保证可复现）

# ── 爆发段检测参数 ─────────────────────────────────────────────────────────
AMP_HIGH_THRESH = 74.0  # 高振幅阈值 (dB)，用于计算高振幅 hit 速率
RATE_BIN        = 5.0   # 速率统计时间箱宽 (s)
RATE_SMOOTH     = 3     # 速率平滑窗口 (bins)
T_EARLY         = 200.0 # 用于估算基准速率的早期时间段上限 (s)
BURST_RATE_MULT = 3.0   # 速率超过基准的倍数即判定为爆发

# ── 声发射事件空间定位参数 ────────────────────────────────────────────────────
EVT_Q_MIN      = 0.5   # 最低定位质量因子（低于此值视为定位不可靠，不显示）
EVT_CLEAN_FRAC = 0.5   # 事件被认定为"干净"所需的最低干净 hits 占比

# Step3 输出目录
STEP3_DIR = os.path.join(RESULT_DIR, 'step3_空间分布')

# ── b 值分析参数 ─────────────────────────────────────────────────────────────
B_WINDOW_N = 200    # 滑动窗口 hits 数（越大越平滑，越小响应越灵敏）
B_STEP_N   = 10     # 滑动步长（hits 数）
B_AMP_MIN  = 40.0   # 振幅完整性下限 (dB)；通常取仪器触发阈值

# ── 绘图参数 ───────────────────────────────────────────────────────────────
CH_COLORS = {
    1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c',
    4: '#d62728', 5: '#9467bd', 6: '#8c564b',
}
SCATTER_S  = 1.5    # 散点大小
SCATTER_A  = 0.40   # 散点透明度
SAVE_DPI   = 200    # 输出图像分辨率
