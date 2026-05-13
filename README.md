# Emotion-Style Controllable LLM via LoRA Interpolation

<p align="center">
  <b>基于 LoRA 插值的情绪风格可控大模型回复生成</b><br>
  <i>Continuous Emotional Control Through Parameter-Space Interpolation of Style-Specific LoRAs</i>
</p>

---

## 核心思想

我们提出一种简单而有效的方法来实现**大模型回复情绪风格的连续可控生成**：

1. **训练阶段**：分别训练 4 个情绪风格的 LoRA 适配器（共情、理性、鼓励、安全）
2. **推理阶段**：在 LoRA 参数空间中进行**线性插值/融合**，将多个风格按任意权重混合
3. **控制界面**：用户通过**滑动条**实时调整各风格权重，观察模型回复如何连续变化

> 这和 Mixture of LoRA Experts (MoELoRA) 的思路一致，但我们提供一个课程级别的简化实现，适合计算资源有限的研究场景。

## 四种情绪风格

| LoRA 适配器 | 风格描述 | 典型表达 |
|:---:|---|:---|
| 🤗 **温柔共情** | 安慰、理解、陪伴 | "我能感受到你的感受，我在这里" |
| 🧠 **理性分析** | 分析问题、给建议 | "让我们系统性地分析这个问题" |
| 💪 **鼓励激励** | 鼓励行动、增强信心 | "你比想象中更坚强，一定能做到！" |
| 🛡️ **冷静安全** | 避免过度情绪，处理负面表达 | "你的感受值得被关注，请优先照顾自己" |

**关键演示**：同一句用户输入，通过改变混合权重可以产生完全不同风格的回复。

## 项目结构

```
.
├── configs/                # 各风格训练配置
│   ├── empathetic.yaml
│   ├── rational.yaml
│   ├── encouraging.yaml
│   └── calm_safe.yaml
├── data/
│   ├── train/              # 训练数据 (JSONL)
│   │   ├── empathetic.jsonl
│   │   ├── rational.jsonl
│   │   ├── encouraging.jsonl
│   │   └── calm_safe.jsonl
│   └── eval/               # 评估数据
│       └── style_comparison.jsonl
├── demo/
│   └── app.py              # Gradio 演示界面（滑动条控制）
├── scripts/
│   ├── train_all.sh        # 训练所有 LoRA
│   ├── generate.sh         # 命令行交互生成
│   ├── run_demo.sh         # 启动 Gradio Demo
│   └── evaluate.py         # 评估脚本（权重网格扫描）
├── src/
│   ├── __init__.py
│   ├── train_lora.py       # 单风格 LoRA 训练
│   ├── interpolate.py      # LoRA 插值核心算法
│   ├── generate.py         # 可控生成推理
│   └── utils.py            # 工具函数
├── notebooks/
│   └── demo.ipynb          # Jupyter 实验笔记本
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 训练四个风格 LoRA

```bash
# 分别训练（每个风格约需 30-60 分钟，取决于数据量和 GPU）
bash scripts/train_all.sh Qwen/Qwen2-1.5B-Instruct data/train
```

或者单独训练某个风格：

```bash
python src/train_lora.py \
    --style empathetic \
    --model_name Qwen/Qwen2-1.5B-Instruct \
    --data_path data/train/empathetic.jsonl \
    --output_dir outputs/lora/empathetic
```

### 3. 命令行交互生成

```bash
# 使用混合权重生成（70% 共情 + 30% 鼓励）
python src/generate.py \
    --lora_paths "empathetic=outputs/lora/empathetic,encouraging=outputs/lora/encouraging" \
    --weights "empathetic=0.7,encouraging=0.3" \
    --input "最近工作压力很大，感觉喘不过气来"
```

### 4. 启动 Gradio 演示

```bash
bash scripts/run_demo.sh
# 打开 http://localhost:7860
```

演示界面提供：
- 四个滑动条实时调整共情/理性/鼓励/安全权重
- 相同输入不同权重的回复对比
- 预设场景快速体验

### 5. 评估

```bash
python scripts/evaluate.py \
    --lora_dir outputs/lora \
    --grid_resolution 5 \
    --output outputs/eval_results.json
```

## 技术方法

### LoRA 参数空间插值

给定 N 个风格 LoRA $\{ \Delta W_1, \Delta W_2, ..., \Delta W_N \}$ 和混合权重 $\{ \alpha_1, \alpha_2, ..., \alpha_N \}$，插值后的 LoRA 参数为：

$$ \Delta W_{interp} = \sum_{i=1}^{N} \alpha_i \cdot \Delta W_i \quad \text{where} \quad \sum_i \alpha_i = 1 $$

**为什么有效？** 研究表明 LoRA 参数在低秩子空间中具有良好的线性结构，不同风格 LoRA 的线性组合可以产生光滑的风格过渡，而不会导致模型退化。

### 与现有工作的关系

| 方法 | 复杂度 | 推理成本 | 本项目 |
|:---|:---:|:---:|:---:|
| 直接 Prompt 引导 | 低 | 1× | 基线 |
| LoRA-MoE (Router) | 高 | N× | 复杂版 |
| **LoRA 参数插值 (本项目)** | 中 | 1× | ✅ |

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{emotion-lora-interp,
  title   = {Emotion-Style Controllable LLM via LoRA Interpolation},
  author  = {Your Name},
  year    = {2025},
  howpublished = {GitHub repository},
}
```

## License

MIT License
