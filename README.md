# EMO-LLM: 大模型情绪计算实验合集

<p align="center">
  <b>SPR-VR 句子级风格控制 · LoRA 插值探索 · RECCON 情绪原因定位</b><br>
</p>

---

## 主项目：SPR-VR — 句子级 Prompt 路由与验证修复

### 一句话

不告诉模型"请 50% 共情、50% 建议"——直接把回复拆成 4 句话，通过外部状态机逐句路由到不同 Prompt Expert，并用独立 Verifier 检查修复。

### 核心结果

```
方法                    MAE↓    Pearson↑  R²↑      ExactMatch↑
──────────────────────────────────────────────────────────────
Prompt-only            0.338    0.212    -0.470    0.053
Structured Prompt      0.312    0.253    -0.317    0.060
SPR                    0.165    0.637    +0.137    0.487
SPR-VR (我们的方法)     0.147    0.706    +0.273    0.507
```

- MAE 比 Prompt-only 降低 **2.3×**
- Pearson 相关性提升 **3.3×**
- Exact Match 提升 **9.5×**（5.3% → 50.7%）
- **零训练，纯推理**，单 GPU 可复现

### 为什么不是 prompt engineering

Structured Prompt 给了完全相同的模板 [E, E, A, A]，但一次生成全部 4 句的 Exact Match 只有 6%。SPR-VR 是 50.7%。**同样的模板，闭环架构使结果差了 8.5 倍。** 创新不是模板，是把生成任务拆成"单句执行 + 外部状态追踪 + 独立验证"的闭环系统。

### 方法架构

```
用户情绪输入
    ↓
[模块一] 风格预算规划器   → 句子功能计划 [E, E, A, A]
    ↓
[模块二] 共享语义计划器   → 内部连贯计划
    ↓
[模块三] 句子级 Prompt 路由 → 逐句调用 Empathy/Advice Expert
    ↓
[模块四] Verifier 检查与修复 → 判别每句类型，不符则重生成
    ↓
最终回复（4 句）
```

### 快速启动

```bash
# 运行完整实验（30 输入 × 5 α × 4 方法）
python scripts/spr_vr_experiment.py

# 结果输出到 log/spr_vr_results.json
```

---

## 子项目二：LoRA 插值风格控制（探索性实验）

### 做了什么

在 Qwen3-1.7B 上训练 4 个情绪风格 LoRA（共情/理性/鼓励/安全），尝试通过参数空间插值实现连续风格控制。

### 核心发现

- 四个 LoRA 权重余弦相似度 0.84~0.90，风格间方差 ≈ 0
- LoRA 训练**削弱**了基座模型的风格区分能力（+0.016 → -0.023）
- ESConv 数据同质化是根本原因

### 结论

在 Qwen3-1.7B + rank=16 LoRA + ESConv 条件下，LoRA 无法有效解耦情绪风格。这个发现直接促成了 SPR-VR 的方法转向。

---

## 子项目三：RECCON 情绪原因定位（探索性实验）

### 做了什么

在 RECCON 数据集上实现 Appraisal-Guided 多信号检索方法，定位对话中导致特定情绪的原因句。

### 核心结果

```
方法                         P@1     R@3     MRR
────────────────────────────────────────────────
Nearest Previous            0.112   0.112   0.112
TF-IDF                      0.194   0.418   0.289
Embedding (Qwen3)           0.194   0.459   0.311
Appraisal (fixed weights)   0.204   0.490   0.328
```

提升 5%（P@1）— 信号存在但不够显著。

---

## 项目结构

```
EMO-LLM/
├── src/
│   ├── interpolate.py          # LoRA 参数空间插值
│   ├── generate.py             # 可控生成推理
│   ├── train_lora.py           # 单风格 LoRA 训练
│   └── utils.py                # 工具函数与风格定义
├── scripts/
│   ├── spr_vr_experiment.py    # SPR-VR 主实验 ★
│   ├── reccon_experiment.py    # RECCON 情绪原因实验
│   ├── experiment_final.py     # 单 LoRA 消融实验
│   ├── experiment_v3.py        # v3 多 LoRA 实验
│   ├── validate_interpolation.py  # LoRA 插值验证
│   ├── evaluate_metrics.py     # 定量指标评估
│   ├── convert_esconv.py       # ESConv → 风格 JSONL
│   ├── train_style.py          # 单卡 LoRA 训练
│   └── train_parallel.sh       # 四风格并行训练
├── configs/                    # 各风格训练配置
├── data/
│   ├── train/                  # ESConv 风格化训练数据
│   ├── train_v2/               # 对比式生成数据
│   └── eval/                   # 测试数据
├── outputs/                    # 训练好的 LoRA 适配器
├── log/                        # 实验日志与指标
├── doc/
│   └── report.md               # 最终课程报告 ★
└── README.md
```

## 引用

```bibtex
@misc{emo-llm,
  title   = {EMO-LLM: Emotion Computing Experiments with LLMs},
  author  = {Your Name},
  year    = {2025},
  howpublished = {GitHub repository},
}
```

## License

MIT License
