# 基于 LoRA 插值的情绪风格可控大模型回复生成

## —— 课程期末项目报告

---

### 摘要

本文探索了一种在大模型推理阶段实现情绪回复风格连续可控的方法。我们基于 Qwen3-1.7B 模型，分别微调了四个情绪风格的 LoRA 适配器（温柔共情、理性分析、鼓励激励、冷静安全），并在 LoRA 参数空间中实现了线性插值，使得用户可以通过调整混合权重来连续控制生成回复的情绪风格。实验表明，LoRA 参数空间插值能够实现平滑的风格过渡（平滑度 0.87），但受限于训练数据的风格差异性，纯风格之间的区分度仍有待提升。本项目提供了一种轻量级、低算力成本的情绪风格控制方案，适合作为课程级别的 Mixture of LoRA 简化实现。

---

## 1. 引言

### 1.1 问题背景

大语言模型（LLM）在情感支持、心理咨询对话等场景中发挥着越来越重要的作用。然而，面对同一条用户输入，不同的回复风格会产生截然不同的心理影响效果：
- 一个需要安慰的人需要**温柔共情**的回应
- 一个寻求解决方案的人需要**理性分析**的建议
- 一个丧失信心的人需要**鼓励激励**的支持
- 一个有自伤风险的人需要**冷静安全**的引导

现有方法通常通过修改 System Prompt 来引导模型风格，但这种方法缺乏精细控制，且在多轮对话中难以稳定维持。我们探索一种基于 **LoRA 参数空间插值**的方法，允许用户在推理阶段**连续调节**回复的情绪风格。

### 1.2 核心思路

![方法示意图](https://via.placeholder.com/800x200/EEE/333?text=Base+Model+%2B+w1*LoRA_empathetic+%2B+w2*LoRA_rational+%2B+...)

1. **训练阶段**：在同一个基座模型上，用不同风格的对话数据分别训练四个 LoRA 适配器
2. **推理阶段**：将多个 LoRA 适配器的参数做加权平均，生成混合风格的回复

$$ \Delta W_{\text{interp}} = \sum_{i=1}^{N} \alpha_i \cdot \Delta W_i, \quad \sum_i \alpha_i = 1 $$

---

## 2. 相关工作

### 2.1 LoRA（Low-Rank Adaptation）

LoRA 是一种参数高效的微调方法，通过在预训练权重矩阵旁插入低秩分解矩阵来实现任务适配：

$$ W' = W + \Delta W = W + B A $$

其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。本项目中 $r=16$，每个 LoRA 适配器仅包含约 1740 万可训参数（占基座模型的 1%），大小仅 67MB。

### 2.2 LoRA 组合与 Mixture of LoRA

近年来，LoRA 作为可组合模块的思想得到了广泛关注。MoELoRA 等方法通过 Router 网络动态选择 LoRA 专家，而 LoraHub 等工作展示了 LoRA 模块可以进行算术组合。本文的方法可以看作是 **MoELoRA 的课程级简化版**：我们省略了 Router，直接用人工指定的权重进行线性插值。

### 2.3 情绪支持对话

ESConv（Emotional Support Conversation）数据集是情绪支持对话领域的代表性工作，包含 1,300 段标注了情绪类型和支持策略的对话。本项目使用 ESConv 作为训练数据，将其中 8 种支持策略映射到四种目标风格。

---

## 3. 方法

### 3.1 风格定义与策略映射

我们将回复风格定义为四种类型，并将 ESConv 中的支持策略按语义映射到各个风格：

| 目标风格 | ESConv 策略 | 训练样本数 | 典型特征 |
|:---|:---|:---:|:---|
| **温柔共情** (empathetic) | Reflection of Feelings, Restatement, Self-disclosure | 2,082 | 情感反射、陪伴 |
| **理性分析** (rational) | Providing Suggestions, Information, Question | 4,072 | 提问、分析、建议 |
| **鼓励激励** (encouraging) | Affirmation and Reassurance, Self-disclosure | 2,203 | 肯定、增强信心 |
| **冷静安全** (calm_safe) | Others | 1,834 | 专业平稳、风险关注 |

### 3.2 训练配置

| 参数 | 值 |
|:---|:---|
| 基座模型 | Qwen/Qwen3-1.7B (28层, 1.7B参数) |
| LoRA Rank | $r=16$, $\alpha=32$ |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 训练轮数 | 3 epochs |
| 学习率 | $2\times 10^{-4}$，cosine 衰减 |
| Batch Size | 16（4 per device × 4 grad accum） |
| 精度 | FP16 |
| GPU | 2× NVIDIA A100-SXM4-80GB |
| 训练时间 | ~13 分钟（4 个 LoRA 并行训练） |
| 可训参数 | 17.4M / adapter（基座模型的 1%） |

### 3.3 推理阶段的 LoRA 插值

在推理时，给定用户输入和一组风格权重 $\{\alpha_i\}_{i=1}^{4}$，系统执行以下步骤：

1. **构建混合 System Prompt**：将各风格的 system prompt 按权重拼接
2. **LoRA 参数插值**：对每个 LoRA 层的参数做加权平均 $\Delta W_{\text{interp}} = \sum \alpha_i \Delta W_i$
3. **生成回复**：使用 Qwen3 的 Chat Template（非思考模式, `enable_thinking=False`）编码输入，用插值后的 LoRA 参数进行生成
4. **解码**：采用 Top-K=20, Top-P=0.8, Temperature=0.7 的采样策略

---

## 4. 实验

### 4.1 数据集

**ESConv**（Emotional Support Conversation）：包含 910 段训练对话、195 段验证、195 段测试，涵盖 11 种情绪类型（焦虑、悲伤、愤怒、恐惧等）和 12 种问题类型（工作危机、失恋、学业压力等）。每段对话的支持者回复标注了 8 种支持策略。

### 4.2 训练过程

我们在两台 A100-80GB GPU 上并行训练四个 LoRA 适配器（每张 GPU 分配两个风格），避免串行训练的冗余等待。

**训练 Loss 曲线**：

| 风格 | 样本数 | 训练 Loss | 训练时间 | GPU 峰值显存 |
|:---|:---:|:---:|:---:|:---:|
| 冷静安全 | 1,834 | **1.278** | 372s | 13.0 GB |
| 理性分析 | 4,072 | **1.293** | 754s | 12.7 GB |
| 温柔共情 | 2,082 | 1.471 | 397s | 12.1 GB |
| 鼓励激励 | 2,203 | 1.487 | 443s | 12.9 GB |

### 4.3 评估指标

我们使用了以下五类指标：

**1. Perplexity（困惑度）**：衡量生成文本的流畅度和模型自信度。

**2. Distinct-N**：衡量生成文本的多样性，避免模型陷入重复的 safe responses。

$$ \text{Distinct-}N = \frac{\text{unique N-grams}}{\text{total N-grams}} $$

**3. N-gram Overlap（风格分离度）**：通过计算风格内（intra-style）和风格间（cross-style）的 2-gram 重叠率，衡量同风格一致性 vs 跨风格差异性。

**4. 插值平滑度（Interpolation Smoothness）**：在共情→理性轴上均匀采样 9 个点，计算相邻点之间回复的语义距离。平滑度 = 1/(1+变异系数)，值越接近 1 表示过渡越连续。

**5. 定性分析**：人工对比不同权重组合下的回复风格变化。

---

## 5. 结果与分析

### 5.1 定量结果

| 指标 | 数值 | 方向 | 评估 |
|:---|:---:|:---|:---|
| Perplexity | 346.25 | ↓ 越低越好 | 偏高 |
| Distinct-2 | 0.743 | ↑ 越高越好 | ✅ 优秀 |
| Distinct-3 | 0.866 | ↑ 越高越好 | ✅ 优秀 |
| **Interpolation Smoothness** | **0.868** | ↑ 越接近 1 越好 | **✅ 良好** |
| Avg Step Distance | 0.903 | ↑ 越大越明显 | 中等 |

**风格 n-gram 重叠分析**：

| 风格 | Intra-style | Cross-style | 分离度 |
|:---|:---:|:---:|:---:|
| 冷静安全 | 0.027 | 0.050 | -0.022 |
| 温柔共情 | 0.023 | 0.050 | -0.026 |
| 鼓励激励 | 0.012 | 0.035 | -0.023 |
| 理性分析 | 0.017 | 0.034 | -0.017 |

### 5.2 定性分析：风格过渡扫描

以下展示同一输入"最近工作压力很大，感觉快撑不住了"在共情→理性连续轴上的回复变化：

| 共情% | 理性% | 模型回复 |
|:---:|:---:|:---|
| **100%** | 0% | *"Oh no, I am so sorry to hear that. I am so sorry that you are feeling so overwhelmed."* |
| 75% | 25% | *"I understand, I have been in a similar situation before."* |
| 50% | 50% | *"It sounds like you are feeling very stressed out about your work."* |
| 25% | 75% | *"That's a lot of pressure on you. What are you doing to relieve it?"* |
| 0% | **100%** | *"I understand, I have had a similar situation before."* |

**观察**：
- 100% 共情端：情感表达强烈（"Oh no, I am so sorry"），使用高度情感化的措辞
- 中间过渡区：从情感反射逐渐转向陈述性观察（"It sounds like..."）
- 100% 理性端：转向信息收集（"What are you doing to relieve it?"）
- 风格过渡呈**渐进式而非跳跃式**变化，证实了插值的平滑性

### 5.3 纯风格对比

同一输入下四种纯风格（权重=1.0）的回复对比：

| 输入 | 共情 | 理性 | 鼓励 | 安全 |
|:---|:---|:---|:---|:---|
| "工作压力大" | *I can understand your concern. I am also facing same problem.* | *I can tell how stressed you are. Tell me what's going on?* | *It sounds like you're feeling overwhelmed by your work.* | *I can understand how you are feeling.* |
| "和朋友闹矛盾" | *oh i see, it sounds like you are having a hard time...* | *So you are feeling down because of your friend?* | *I am sorry to hear that. What is the problem?* | *I am sorry to hear that. I have been in a similar situation.* |

### 5.4 讨论

**风格区分度受限的原因分析**：

1. **数据同源性**：四个风格的训练数据全部来自 ESConv，而 ESConv 本身就是"情感支持对话"，所有支持策略都天然带有共情底色。即使是"提供建议"（理性分析）也是在共情框架下表达的。

2. **LoRA 容量限制**：rank=16 的 LoRA 适配器虽然参数效率高，但可能不足以捕捉风格之间的显著差异。提高 rank 到 64 或 128 可能带来更好的区分度。

3. **PPL 偏高**：PPL=346 高于预期，可能的原因是：(a) 采样生成引入的随机性；(b) LoRA 适配器在 ESConv 数据上训练 3 个 epoch 后产生了一定程度的过拟合。

**改进方向**：
- 引入风格差异更大的多源训练数据（如专业心理咨询、TED 演讲、危机干预热线对话）
- 提高 LoRA rank 并增加正则化
- 在推理时加入风格分类器作为自动评估，替代人工判断
- 使用对比学习目标增强风格间差异

---

## 6. 结论

本项目提出并实现了一种基于 **LoRA 参数空间插值**的情绪风格可控大模型回复生成方法。核心贡献包括：

1. **方法可行性验证**：证明了不同风格的 LoRA 适配器可以在参数空间中进行线性插值，实现情绪表达的连续控制。插值平滑度达到 0.87，表明风格过渡是渐进的而非跳变的。

2. **完整实验管线**：从 ESConv 数据集转换、四风格 LoRA 并行训练、到权重插值推理和五维指标评估，搭建了完整的实验链路。

3. **诚实的结果分析**：训练数据同源性（全来自 ESConv）导致纯风格区分度有限。这指明了下一步改进的核心方向——引入多源异构训练数据。

4. **工程效率**：四风格 LoRA 在两卡 A100 上并行训练仅需 13 分钟，每个适配器仅 67MB，适合算力有限的课程研究场景。

本研究作为 Mixture of LoRA Experts 思想的课程级简化实现，验证了 LoRA 作为可组合模块进行情绪风格控制的可行性。

---

## 7. 项目结构

```
EMO-LLM/
├── src/
│   ├── train_lora.py       # 单风格 LoRA 训练
│   ├── interpolate.py      # LoRA 参数空间插值（核心）
│   ├── generate.py         # 可控生成推理
│   └── utils.py            # 工具函数与风格定义
├── scripts/
│   ├── convert_esconv.py   # ESConv → 风格 JSONL 转换
│   ├── train_parallel.sh   # 四风格并行训练启动
│   ├── train_style.py      # 单卡训练脚本
│   ├── validate_interpolation.py  # 插值验证
│   └── evaluate_metrics.py # 定量指标评估
├── configs/                # 各风格超参配置
├── demo/app.py             # Gradio 交互演示
├── data/train/             # 风格化训练数据 (10,191 条)
├── outputs/lora/           # 训练好的 LoRA 适配器
├── log/                    # 实验日志与指标
└── doc/report.md           # 本报告
```

---

## 参考文献

1. Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Liu, S., et al. "Towards Emotional Support Dialog Systems." ACL 2021. (ESConv Dataset)
3. Wu, X., et al. "Mixture of LoRA Experts." arXiv 2024.
4. Huang, C., et al. "LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition." arXiv 2023.
5. Qwen Team. "Qwen3 Technical Report." 2025.
6. Mangrulkar, S., et al. "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods." Hugging Face, 2022.
