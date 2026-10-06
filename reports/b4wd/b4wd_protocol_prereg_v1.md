# B4-WGAN / B4-扩散（B4WD）协议预注册 v1 —— 已冻结

- 日期：2026-09-28（草案同日冻结）；阶段：B4′ 机制探究第二项
- **状态：已冻结（2026-09-28，用户"继续完成"指令批准）。判读规则
  §4/§5 与 §2 全部实现约束自此不可移动；任何正式训练/生成作业必须
  晚于本冻结。§3 代码锚与冒烟记录属实现期回填项，不属判读规则**
- 直接前置：B4R 层 2 终判（`reports/b4r/b4r_protocol_prereg_v1.md` §8）——
  x̄ = 11.098W ≤ 15.64，**生成器模糊为主因**成立，预注册后续动作即本项
- 性质：判读规则先于运行写入；不放宽、不事后改口径；B4R 已定案内容不可修改
- 事实底稿：2026-09-28 四路侦察（基元管线 / 整周期生成器迁移 / sbatch 先例 /
  锚点数字，全部结论带 file:line 溯源）
- 修订记录：初稿经 20-agent 对抗预注册审查（7 维度审查 + 逐条反驳式
  核verify），**11 项 CONFIRMED 发现**（上界噪声边际不足 1.5W<SE 2.16W、
  WGAN mask 语义未定义、mask 不能落 scripts 层、identical_pairs 指标口径
  失实 + 种子碰撞通道未预声明、坍缩门在克隆管线中被静默放行、diversity
  只覆盖前 200/246 条、de_inputs 克隆清单漏 S2/S3 heredoc 字面量、S0a
  锚 29e4a92 不锁 B4WD 代码、坍缩门"候选/讨论"措辞留事后裁量、checkpoint
  选取准则未预注册等）与经采纳的 minor 项于 2026-09-28 并入本版；
  2 项被驳回（训练期 one-hot 恒零不破 parity 断言——其事实以披露方式
  记入 §2；WGAN masked 训练可实现——其语义以预声明方式记入 §2）。
  **并入时点先于任何 B4WD 运行**，未执行过任何门判读，不构成事后改口径。

## 1. 动机与问题

B4R 证明：组合器不动、基元换 real donor，val MAE 从 19.43±3.74 掉到
11.10±1.07。剩余问题：**可训练的基元生成器（非 donor 直取）能恢复这段
8.33W 差距的多少？**

| 基元来源 \ 组合器 | 经验路径 + 直接拼接 |
|---|---|
| CVAE 生成（B4 现状） | 19.43±3.74（判负） |
| 真实状态段（B4R） | **11.10±1.07（上界锚）** |
| 条件 WGAN（B4WGAN） | **本项** |
| 条件扩散（B4DIFF） | **本项** |

先验理由：B3WGAN/B3DIFF 在**整周期级**已证 WGAN/扩散能过质量门且下游打平
B2@r2.0 平台、显著优于 B2@r0.5——B3WGAN 三种子 9.4826/9.385/11.945
（jobs 4160/4170/4171，x̄=10.27±1.45，F1 0.823 全场最高）；B3DIFF
10.5489/8.994/11.666（4161/4172/4173，x̄=10.40±1.34，F1 0.776）；配对
bootstrap B3-G Δ=−1.57W、B3-D Δ=−1.44W（进展报告 §8.1）。本项检验其
"锐度"优势能否在**状态段级**复现。B3-V 教训（整周期 CVAE 质量门 PASS 但
下游 30.00±3.23 垫底）预声明：**生成质量门只作废式门，判定统计量一律是
下游 val MAE**。

## 2. 干预定义（组合器与 rng 序列零改动，只换基元解码器后端）

- `PrimitiveComposer` 的 `primitive_source` 已有 {"model","donor"}；本项
  扩展 model 后端：{"cvae","wgan","diffusion"}（donor/real 路由不动，
  B4R 复现性保持）。新增 arm 标签 **B4WGAN / B4DIFF**（镜像
  validation_master_table 的 B3WGAN/B3DIFF 命名；prepare 脚本标签规则
  合法：字母数字+下划线、不含 "test"）。
- **不变量（全部沿 B4R 已闭合的硬门模板）**：
  1. 路径模型、donor 选取的 numpy rng 调用序列（`generate_cycle` 仅三处
     消耗：n_states 抽取 `primitive_cvae.py:216`、sample_path `:219`、
     donor 选取 `:230`）、段长（=donor 长）、`np.concatenate` 拼接、
     MAX_PATH_STATES=12（`:24`）全部不动；
  2. **等价性断言照用**：同 generation seed=17 下，B4WGAN/B4DIFF 的每条
     记录 `state_path` + 逐段 `target_samples` 与 `b4_s17_4133` 归档
     逐条相等（B4R 已证路径/donor/段长与解码器无关；**新解码器只允许
     消耗 torch rng**——WGAN 沿用 torch.manual_seed(rng_seed)+randn 取
     latent 后前向，扩散以 rng_seed 定初始噪声、整周期 _SamplingAdapter
     先例；任何 numpy rng 消耗即破等价，作废）。推论：两臂之间
     path/donor/段长同样逐条相等——B4/B4WGAN/B4DIFF 三方唯一自由度
     = 段波形内容；
  3. 条件信息集与 B4 CVAE 基元完全一致（侦察核实）：条件向量 = 4 维
     归一化统计（length / mean power / energy / peak，取自被选中 donor
     波形）+ 状态 one-hot（`primitive_cvae.py:126-135` segment_condition）；
     **长度以归一化标量进条件、无长度桶 one-hot**——训练侧
     build_segment_conditions（`:104-123`）的 6 桶 LengthBucketizer 与
     normalizer 只作数据准备/长度选择，不进条件向量，原样复用。
     **披露（审查驳回项的事实残留）**：B4 既有实现中训练条件的 one-hot
     通道恒为全零（build_segment_conditions 不写入 labels）、解码时才
     置位——该 train/decode 条件错位被两新臂**原样继承**（与 B4 同构，
     不构成臂间差异，但结果解释时须知状态信息在训练中仅经 4 统计间接
     进入）。架构（CVAE→WGAN/扩散）是唯一差异；
  4. 输出后处理到瓦特域同口径：截到 donor 长、clip≥0 不变。功率尺度：
     **B4WGAN 沿用 CVAE 基元的全局峰值 q99 scale**（
     train_full_cycle_generator.py:58-60，语义与 B4 逐位一致）；
     **B4DIFF 按 unit-variance 训练要求改用 std 尺度**（整周期 d83f200
     先例），并**按状态分别拟合**（每状态段 std 的 q95、下限 1.0——
     状态间 median 功率 153→1866W 差 12 倍，全局单一 std 会重演低功率
     状态信噪比失衡）；生成侧峰值截断按状态上限（该状态 donor 峰值
     max，d83f200 方法级截断的段级化）。两处尺度差异为预声明的架构
     适配，非事后口径。
- **实现约束（违者不得提交）**：
  - **落点分层（审查修正）**：std 尺度拟合与峰值截断是数据缩放/后处理
    性质的修复，落 **scripts/ 层**（d83f200 先例同性质）；**mask 属训练
    循环内部**，落 **src/generation 新增段级训练函数**（见下），不得
    复制整周期循环到脚本层了事；
  - 整周期管线的 std 尺度与峰值截断修复在 scripts/ 层而非 src/generation
    模块内——迁移必须显式带入段级路径，否则重演 impossible_peak FAIL
    （扩散整周期前科 job 4136：峰值尺度致 std≈0.13、noise_mse 恒 1.005、
    采样残差 ×3975 变野峰值）；
  - **段级训练必须带 mask，两臂语义分别预声明**：
    - 扩散臂：masked noise-MSE——mask 从 pad_and_mask 保留（整周期
      `full_cycle_diffusion.py:124` 丢弃 mask 的做法不迁移），损失只在
      有效样本点上平均（参照 elbo_loss 的 valid-mask 写法，
      full_cycle_cvae.py:145-152）；落点为 src/generation 新增
      `train_primitive_diffusion`；
    - WGAN 臂：real 与 fake 波形均按有效段长**掩零后进 critic**，
      梯度惩罚的插值在掩零后的 real/fake 之间进行，w_distance 监视器
      在掩零输入上计量；生成器输出先出全长再按 real 的 mask 掩零
      （WGAN-GP 无逐样本重构损失，"masked loss"以其输入侧掩码语义
      实现——审查发现 #2 的定案）；落点为 src/generation 新增
      `train_primitive_wgan`；两臂掩码行为各配单测锁定；
  - 状态段长展布（median 324/3294/1266/630s）远大于整周期，短段 pad
    不设防会被主导——上条 mask 即为防线的机判形式；
  - wave_length = 基元桶化器最大桶长（8 的倍数），与 CVAE 基元同构；
    checkpoint 保存 wave_length/bucketizer/normalizer/n_states/power_scale；
  - 生成期单进程、无并行 DataLoader worker 消耗全局 torch RNG
    （`_decode_primitive` 每段 torch.manual_seed 重置全局种子，
    `:191-207`——并发会破解码确定性）；
  - 生成命令固定 **PYTHONHASHSEED=0**：rng_seed 公式含加盐
    `hash(synthetic_cycle_id)`（`:248`），不固定则解码种子跨进程不可
    复现。**补充（审查 minor）**：str hash 的确定性以解释器构建为界，
    故 run 目录记录服务器 Python 版本，且 cycles manifest 落盘每段
    rng_seed（复现性可哈希门机检，不隐式依赖环境一致）。B4/B4R 归档
    不受影响（B4R 不经解码器；B4 波形以归档为准，不允许也不需要再生成）。

## 3. 运行参数

- 状态库：冻结 v1（k4，1930 块/493 周期），哈希前置同 B4R
  （inventory `dfa3e506…` / waveforms `5b81d524…`，服务器端复验，
  不等即停）。
- **代码锚（审查新增）**：正式作业的服务器 HEAD 必须包含 B4WD 实现
  提交（解码后端分派 + 段级训练函数 + 脚本层尺度修复 + 管线克隆），
  哈希于 B4WD 实现提交合入时回填：
  `bfa3df9d6967996865ede753bf630c454c94766f`（2026-09-29 已回填）；
  克隆管线 S0a 的祖先锚（原 29e4a92）已同步替换为该哈希——否则 S0a
  对含/不含 B4WD 代码的 HEAD 一律放行，冻结纪律无机检锚点。
- 基元生成器训练：先本地 CPU 冒烟（小 count；架构/超参迭代**只允许在
  冒烟期**），冻结超参后服务器正式训练各一次：
  - B4WGAN：epochs 150、n_critic 5、gp_weight 10、Adam(1e-4,
    betas 0.5/0.9)、latent 16、width 32（B3-G 先例 slurm/b3_wgan.sbatch
    + full_cycle_wgan.py 默认）；
  - B4DIFF：epochs 300、扩散步 500（线性 1e-4→0.02）、width 32
    （B3-D 先例 slurm/b3_diffusion.sbatch + full_cycle_diffusion.py
    默认）；
  - **沿用 B3 超参数值（非等量更新预算——1930 段 vs 493 整周期，同
    epochs 下优化器步数约 4 倍；审查 minor 措辞修正）**；不另行调参
    （调参=超参搜索，§6 范围外）；
  - **checkpoint 选取准则（审查新增，预注册）**：正式生成一律使用
    末轮（第 150/300 epoch）checkpoint；**禁止按任何监视曲线**
    （w_distance / noise_mse / 质量报告）**挑轮**——挑轮即看到训练
    结果后再定生成器。末轮触发作废门 ⇒ 只能按 §4 记负面结果或发新
    协议版本；
  - 训练种子 17（B4 sbatch 先例：训练与生成共用 PSLG_SEED）；
  - 资源：各 1×RTX3090，时限 B4WGAN 1 天 / B4DIFF 2 天（B3 sbatch
    同规格；不采用 A6000 模板——当前分区只有 3090 可用）；
  - 前置：服务器 GPU 容量恢复（2026-09-28 分区现状 h102 down / h103 卡
    不可见 / h104 恢复，见 B4R 执行记录；管理员报修中）；
  - 监视（诊断性证据，不设门）：WGAN w_distance/gradient_penalty 历史
    **按状态分桶记录**（全 batch 混状态标量会掩盖单状态坍缩——审查
    minor）；扩散 noise_mse 曲线（欠训→采样尖峰的已知链条）。
- **冒烟期必查项（审查新增）**：扩散各状态在尺度化后 std 接近 1 且
  noise_mse 持续下降（防状态内 std 重尾使 q95 高于典型 std、或低功率
  状态整体 std<1W 触发下限——两者都会重演 4136 信噪比失败）；若需
  尺度重设计（如改分位数或按段 scale），属架构适配、限冒烟期内定案。
- 生成：count=246，generation seed=17，sample_seconds=6（对齐
  b4_s17_4133 等价基线），PYTHONHASHSEED=0；每臂各自 cycles + 双报告。
  **B4DIFF 生成执行位置（审查新增）**：以 `--device cuda` 的短 GPU
  作业执行（500 步 × 每周期多段 × 246 周期 ≈ 千次采样，不落在登录
  节点 CPU 上；generate_primitive_cycles.py 的 --device 默认 cpu，
  克隆管线 S1 必须显式传参）。等价断言只锁 numpy rng，不受设备影响。
- 放置：`--envelope-samples 2372` 钉住（同 B4R），starts 断言 246/246。
- de_inputs：**每臂独立** v1 补丁目录 `de_inputs_r0p5_b4wgan` /
  `de_inputs_r0p5_b4diff`（侦察定案 + 审查修正：S3 脚本对已存在 INPUTS
  目录即 abort，独立目录规避互踩；**哈希基线 = 冻结 v1（de_inputs_r0p5）**
  ——与 v1 同名 23 文件逐字节相等 + manifest 仅新增本臂；S3 机判断言
  extra 集合 = {B4WGAN} / {B4DIFF} 单元素）。
- 下游：`c3_seq2point.sbatch`，arm=B4WGAN/B4DIFF，训练种子 17/42/73
  复用各自 de_inputs（B4/B4R 同机制：只换训练种子、生成集不变），
  batch 128、200 steps/epoch、max-epochs 30、patience 5、
  val MAE = 20k 监控窗口口径。

## 4. 质量门与记忆化口径（先声明，不放宽）

- `evaluate_synthetic_quality` 照跑：impossible_peak FAIL 不可豁免
  （扩散整周期前科：欠训尖峰顶到截断上限是已知失败模式——大量样本
  顶到上限=欠训信号，不是过门信号）。**B4DIFF 语义注记（审查 minor）**：
  按状态峰值截断后峰值构造上不可能越限，该门对 DIFF 臂退化为
  "贴上限占比"欠训间接信号——贴上限占比与截断触发率作为单独证据列
  记录（两臂门语义不对称属预声明，非事后口径）。
- duration WARN 预期与 B4/B4R 逐位同构（路径+donor 长度相同，B4R
  out_of_duration=102/246 与 B4 归档逐位一致）；WARN 接受记录必须在
  下游 sbatch 提交前书面留痕（接受人/指标/数值/理由，B4R 惯例）。
- **diversity 指标口径（审查修正，先声明）**：identical_pairs /
  min_pairwise / mean_pairwise 由 diversity_index 计算——**默认仅取
  前 max_cycles=200 条**（246 条中截 46 条；B4 归档 sampled_cycles=200
  为证）、每条线性重采样 256 点、L2 归一化后 d<1e-6 判 identical
  （即"形状空间近同"而非"位级同波形"）。**本项要求覆盖全 246 条**：
  质量评估调用传 max_cycles≥246；若 CLI 未暴露该参数，S1b 机判段对
  全量记录按同口径复算 identical_pairs 与 min_pairwise。
- **坍缩门（零裁量作废式，审查重写）**：
  - **identical_pairs ≥ 1（全 246 条口径）⇒ 本次运行作废 + 书面查因，
    无豁免**。查因只区分机理、不改变作废后果：(i) 生成器坍缩（b3_wgan
    先例：记负面结果，不换参重跑）；(ii) rng_seed 公式 `hash(id)%1_000`
    桶碰撞 × 同 (path, donor) 序列 ⇒ 逐位同波形的确定性重复（246 id
    期望约 30 对种子哈希同桶，与生成器质量无关；修复路径只能是新协议
    版本，如换确定性 id 散列 crc32/sha1）。触发后的讨论只能在"作废 +
    何种机理 + 下一步提案"内进行，**不得判定通过**；
  - **min_pairwise（全量口径）< 1e-3 ⇒ 同款作废查因门**（覆盖"同形状
    不同尺度"坍缩通道——归一化距离对其不敏感；健康参照 B4 归档
    min=0.0257，两个数量级余量）；
  - mean_pairwise 记录为描述性证据，参照带 B4 0.9317 / B4R 0.9840 /
    B3-V 欠训 0.4969；不设数值门。
- **机判承接（审查新增，堵静默放行）**：克隆管线 S1b 机判段新增断言
  `identical_pairs == 0` 与 `min_pairwise ≥ 1e-3`（≥1 / <1e-3 即
  exit 1）；**期望 WARN 集合改为 {duration_distribution}**（diversity
  预期 PASS——B4R 管线的 `{'duration_distribution','diversity'}` 断言
  是 donor 生日碰撞的追认集合，克隆不改会在健康运行上误杀、在坍缩
  运行上放行）。
- 记忆化审计：**无 B4R 式定义性复刻豁免**——B4WGAN/B4DIFF 输出的是
  生成波形而非 donor 直取，**exact_duplicate > 0 一律真 FAIL**；
  replication_rate ≤ 0.01 照跑（B4 归档先例 1/246=0.4% 过门）。
  （段级相似是条件生成的定义属性，审计仍是整周期形状距离粒度，口径同前。）
- 不重跑条款：冒烟期后任何再训练/再生成（换 seed、改超参、改代码）
  都要求新协议版本 + 书面理由，无未经记录的重试；节点 GPU 故障死亡
  作业（零训练结果）不构成重跑（B4R 5345–5349 先例，证据目录保留）。

## 5. 判读规则（预注册；阈值以两极锚定）

锚点（validation_master_table.json 精确值）：**B4 = 19.4304±3.7382**
（15.3872/22.761/20.143，CVAE 下界）；**B4R = 11.098±1.07**（real 上界）；
可恢复空间 = 8.3324W；参考锚 B2@r0.5 = 11.8380±1.6051、B2@r2.0 =
10.3781±0.0814、**B3WGAN = 10.27±1.45**（9.4826/9.385/11.945）、
**B3DIFF = 10.40±1.34**（10.5489/8.994/11.666）。
两臂**各自独立判读**，无联合检验。

注：两 B3 整周期锚（10.27/10.40）均落在下档区间内——若段级继承整周期
质量，预期落点即在下档；本项检验的正是组合器 + 段级条件化能否保持该
水平。此注记不改变判读规则，仅说明下档并非宽松线。

**噪声边际（审查修正：不对称定标，按各锚种子噪声）**：
- 下界 12.60（对 B4R 锚，种子 sd 1.07 → x̄ SE≈0.62W）：边际 **1.5W**
  （沿 B4R 惯例）；
- 上界 17.93（对 B4 锚，种子 sd 3.74 → x̄ SE≈2.16W）：边际 **2.2W**
  （= 3.74/√3 取整；1.5W < 该锚零分布标准误，不足以覆盖——审查发现 #1，
  原 1.5W 一刀切作废）。
边际规则（**强制限定，非可选措辞**）：|x̄ − 边界| < 对应边际 ⇒ 档位
结论必须带"阈值噪声带内"限定，并写明不能排除的另一档（下界带内：
不能排除"与 real 上界不可区分"；上界带内：不能排除"与 B4 不可区分"）。

### 层 1：s17 初筛（方向预览，非终判；单种子限定）

x = 该臂 s17 val MAE。x ≤ 12.60 → 强恢复信号；x ≥ 17.93 → 无恢复信号；
其间灰区。只决定是否提前报告方向。阈值与层 2 完全同数（审查 minor
统一）；边际限定同样适用（下界侧 1.5W / 上界侧 2.2W）。

### 层 2：三种子终判（x̄ = mean(s17,s42,s73)，每臂独立）

| x̄ 落点 | 终判 | 后续动作 |
|---|---|---|
| x̄ ≤ 12.60（= B4R 11.098 + 1.5W） | **与 real 上界不可区分：基元生成器可替代 donor 直取** | 路线成立；如需进 test 另发新版本冻结 |
| 12.60 < x̄ < 17.93（= B4 19.4304 − 1.5W） | **部分恢复**：按恢复比例 ρ 报告，方向结论"架构显著影响基元质量但未达 real" | 视 ρ 区间与两锚距离定性讨论，不做二选一 |
| x̄ ≥ 17.93 | **无恢复：与 B4 不可区分，架构换不掉模糊性** | B4WGAN/B4DIFF 各自判负（单训练种子无恢复信号）；"基元生成器路线整体关账（三架构）"的结论须第二训练种子复现或新版本预注册后方可书写（审查 minor：单训练实例不过度外推） |

- ρ（审查 minor 精确化）：ρ = (19.4304 − x̄)/8.3324（精确锚点），报告
  须附区间 ρ ± 2·(本臂三种子 sd/√3)/8.3324；定性讨论不得引用区间外
  的 ρ 点值。
- 12.60/17.93 两界与两 B3 整周期锚自然相容（均 < 12.60）；**定稿后
  不再移动**。
- 机制证据（描述性，不设门）：peak 分位落点（B4 q50=1929.8 → B4R
  q50=3167.5，real q50=2998.0；两臂落点记录峰值恢复程度）。B3-V
  教训重申：peak/质量门均不替代下游 val MAE 判定。
- 层 2 管辖层 1；不得据本结果修改 B4/B5/B4R 任何已判负/已终案记录。

## 6. 范围外（明确不做）

- 修改 B4R/B4/B5 定案；B4WGAN/B4DIFF 进 test（需新版本冻结）。
- B5-real（真实基元 × HSMM）仍范围外。
- 基元生成器超参搜索（冒烟期外）；多训练种子 × 多下游种子的全因子；
  WGAN 崩溃后的换参重试（只记负面结果，b3_wgan 先例）。

## 7. 产物与证据（计划）

- 代码接入点（侦察行号，含审查修正）：
  - `src/generation/primitive_cvae.py:191-207` `_decode_primitive`
    后端分派——WGAN：`generator(latent, condition)` 前向（签名同
    decode，full_cycle_wgan.py:70-73）；扩散：
    `GaussianDiffusion.sample(denoiser, shape=(batch, 1, wave_length),
    cond, device)`（full_cycle_diffusion.py:94-109，**4 个参数含
    device、shape 为三维**——审查修正；torch.manual_seed(rng_seed)
    定初始噪声）；
  - `src/generation/` 新增段级训练函数 `train_primitive_wgan` /
    `train_primitive_diffusion`（masked，按 §2 掩码语义；不复制整周期
    循环到脚本层）；
  - `scripts/generate_primitive_cycles.py:39-61`：load_checkpoint 按
    route 加载 generator/denoiser；**:94 route 白名单扩展为
    {primitive, primitive-wgan, primitive-diffusion}**；
    **`src/generation/primitive_cvae.py:271-272` route 标签按后端写
    B4WGAN/B4DIFF**（现对一切 model 后端写死 "B4"——审查 minor）；
  - `scripts/train_full_cycle_generator.py:244-259`：新增
    primitive-wgan / primitive-diffusion 训练分支（数据准备整体复用
    build_segment_conditions）；
  - std 尺度/峰值截断两处段级适配落脚本层（§2 落点分层）；
  - cycles manifest 落盘每段 rng_seed（§2 PYTHONHASHSEED 补充）。
- 单测：新增段级 WGAN/扩散训练+采样+等价性测试（现有测试只覆盖整周期
  WGAN/扩散与 CVAE 基元；test_full_cycle_wgan.py /
  test_full_cycle_diffusion.py 为模板）。关键断言：新后端同 seed 下
  `state_path`/`target_samples` 与 cvae 后端逐条相等；**两臂掩码行为
  （real/fake/GP 掩零、masked noise-MSE 只在有效点平均）；段级扩散
  训练/采样在 cuda 设备可跑（fd3a657 时间嵌入设备跟随的显式回归
  保护）**。
- 管线：`b4r_server_pipeline.sh` 克隆改名（**审查补全清单**）：
  - 变量/标签：RUN_DIR、PLACED、INPUTS、`B4R=`→`B4WGAN=`/`B4DIFF=`、
    PSLG_ARM、run 前缀 `s2p_B4R_`→对应臂；
  - **S0a 祖先锚 :21**：29e4a92 → B4WD 实现提交哈希 `bfa3df9d…766f`
    （§3 代码锚，2026-09-29 已回填）；
  - **S1 命令**：改训练模型模式 `--checkpoint-dir`；B4DIFF 显式
    `--device cuda` 且以 GPU 作业执行（§3）；
  - **S1b 机判**：期望 WARN 集合改 `{duration_distribution}`；新增
    `identical_pairs == 0`、`min_pairwise ≥ 1e-3` 断言；diversity
    覆盖全 246 条（max_cycles≥246 或全量复算）（§4）；
  - **S1c 等价断言保留** vs b4_s17_4133（非 donor 特有，rng 序列断言
    对生成后端同样适用）；
  - **S2 heredoc :114**：`new["arms"]["B4R"]` → 对应臂键（独立于
    `B4R=` 前缀替换的字面量——审查发现）；
  - **S3**：extra-arm 行与机断 `extra == {"B4R"}` → 单元素对应臂；
    **heredoc :141-142 硬编码 `v1dir="de_inputs_r0p5"` /
    `newdir="de_inputs_r0p5_b4r"`** → newdir 改对应臂目录、v1dir 不变
    （哈希基线 = 冻结 v1，§3——审查发现）。
- 硬门清单（作废式，逐臂过）：①库哈希 + 代码锚（§3）②等价断言
  （vs b4_s17_4133，246/246）③starts 246/246（envelope 2372）
  ④de_inputs 哈希门（vs v1 同名 23 文件逐字节相等 + manifest 仅新增
  本臂）⑤双门（impossible_peak 不可豁免；exact=0 无豁免；
  **identical_pairs≥1 或 min_pairwise<1e-3 ⇒ 作废 + 书面查因，无豁免**
  ——机判承接见 §4/§7 克隆清单）。

## 8. 结果与终判（回填槽，冻结后运行）

（空——按层 2 口径回填每臂三种子表、x̄、ρ（含区间）、终判与后续动作）
