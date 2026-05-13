"""Gradio demo: interactive style interpolation via sliders.

Single user input → adjust sliders for empathy/rational/encouraging/calm → see how the response changes.

Run:
    python demo/app.py --lora_dir outputs/lora --model_name Qwen/Qwen2-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interpolate import InterpolatableLoRA
from src.generate import generate_with_interpolation
from src.utils import STYLE_NAMES


STYLE_KEYS = ["empathetic", "rational", "encouraging", "calm_safe"]
STYLE_LABELS_ZH = {
    "empathetic": "温柔共情 🤗",
    "rational": "理性分析 🧠",
    "encouraging": "鼓励激励 💪",
    "calm_safe": "冷静安全 🛡️",
}

CSS = """
.gradio-container { max-width: 900px; margin: 0 auto; }
.slider-row { margin: 8px 0; }
.response-box textarea { font-size: 16px !important; line-height: 1.6 !important; }
.history-box textarea { font-size: 13px !important; }
"""


class StyleController:
    """Manages the model and tracks style interpolation state."""

    def __init__(self, model_name: str, lora_dir: str, device: str = "auto"):
        self.model_name = model_name
        self.lora_dir = Path(lora_dir)
        self.device = device
        self.wrapper = None
        self.tokenizer = None
        self.loaded = False
        self.history: list[dict] = []

    def load(self):
        if self.loaded:
            return
        print(f"Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=self.device,
        )

        # Discover LoRA adapters in the directory
        lora_paths = {}
        for style in STYLE_KEYS:
            style_dir = self.lora_dir / style
            if style_dir.exists() and (style_dir / "adapter_model.safetensors").exists() or \
               (style_dir / "adapter_model.bin").exists():
                lora_paths[style] = str(style_dir)

        if not lora_paths:
            print(f"Warning: No LoRA adapters found in {self.lora_dir}. Using dummy mode.")
            self.loaded = True
            return

        print(f"Found LoRA adapters: {list(lora_paths.keys())}")
        self.wrapper = InterpolatableLoRA(
            base_model=model,
            lora_paths=lora_paths,
            interpolation_mode="weight",
        )
        self.loaded = True
        print("Model loaded successfully.")

    def generate(
        self,
        user_input: str,
        w_empathetic: float,
        w_rational: float,
        w_encouraging: float,
        w_calm: float,
        temperature: float,
    ) -> tuple[str, str]:
        if not user_input.strip():
            return "", self._render_history()

        if not self.loaded:
            self.load()

        weights = {
            "empathetic": w_empathetic,
            "rational": w_rational,
            "encouraging": w_encouraging,
            "calm_safe": w_calm,
        }

        if self.wrapper is not None:
            response, info = generate_with_interpolation(
                self.wrapper,
                self.tokenizer,
                user_input.strip(),
                weights,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
            )
        else:
            # Dummy mode for demo without trained LoRAs
            response = self._dummy_generate(user_input, weights)
            info = {"weights": weights, "style_labels": {k: STYLE_NAMES[k] for k in weights}}

        # Build style breakdown text
        breakdown = " | ".join(
            f"{STYLE_LABELS_ZH.get(k, k)}: {weights[k]*100:.0f}%"
            for k in ["empathetic", "rational", "encouraging", "calm_safe"]
        )

        # Add to history
        entry = {
            "input": user_input,
            "response": response,
            "weights": weights,
            "breakdown": breakdown,
        }
        self.history.append(entry)

        return response, self._render_history()

    def _dummy_generate(self, user_input: str, weights: dict) -> str:
        """Generate a plausible response for demo without trained models."""
        top_style = max(weights, key=weights.get)
        responses = {
            "empathetic": f"我能感受到你此刻的心情。{user_input}——这件事确实不容易，请允许自己有这样的感受。我会一直在这里陪着你。",
            "rational": f"我们来梳理一下你提到的情况。关于「{user_input}」，可以从以下几个角度分析：第一，明确核心问题是什么；第二，列出可行的选项；第三，评估每个选项的利弊。建议你先从最可控的部分入手。",
            "encouraging": f"你能说出来，已经是非常勇敢的一步！面对「{user_input}」，我相信你有足够的能力去应对。每一个挑战都是成长的机会，加油，你一定可以做到！",
            "calm_safe": f"感谢你的分享。关于你提到的「{user_input}」，请记住：你的安全感和心理健康是第一位的。如果你感到不适，可以随时暂停或寻求专业帮助。我在这里，以平稳的方式陪伴你。",
        }
        return responses.get(top_style, responses["calm_safe"])

    def _render_history(self) -> str:
        if not self.history:
            return "*（暂无对话历史）*"
        lines = []
        for i, entry in enumerate(self.history[-5:]):  # last 5
            lines.append(f"**[{i+1}] 用户:** {entry['input'][:60]}...")
            lines.append(f"**风格:** {entry['breakdown']}")
            lines.append(f"**回复:** {entry['response'][:120]}...")
            lines.append("---")
        return "\n".join(lines)

    def clear_history(self):
        self.history = []
        return "", self._render_history()


def create_demo(controller: StyleController):
    def on_generate(*args):
        return controller.generate(*args)

    def on_clear():
        return controller.clear_history()

    with gr.Blocks(css=CSS, title="情绪风格 LoRA 插值 - 可控生成演示") as demo:
        gr.Markdown(
            """
            # 🎭 情绪风格 LoRA 插值：可控大模型回复生成

            输入同一句话，通过滑动条调整 **共情 / 理性 / 鼓励 / 安全** 四个维度的权重，
            观察模型回复如何实时变化。

            **核心思想**: 不同情绪风格的 LoRA 适配器在参数空间中连续插值，实现情绪表达的精细控制。
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="💬 你的输入",
                    placeholder="例如：最近工作压力很大，感觉喘不过气来...",
                    lines=3,
                )

                with gr.Row(elem_classes="slider-row"):
                    s1 = gr.Slider(0, 1, value=0.5, step=0.01, label=STYLE_LABELS_ZH["empathetic"])
                    s2 = gr.Slider(0, 1, value=0.5, step=0.01, label=STYLE_LABELS_ZH["rational"])
                with gr.Row(elem_classes="slider-row"):
                    s3 = gr.Slider(0, 1, value=0.5, step=0.01, label=STYLE_LABELS_ZH["encouraging"])
                    s4 = gr.Slider(0, 1, value=0.5, step=0.01, label=STYLE_LABELS_ZH["calm_safe"])

                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="🌡️ 温度")

                with gr.Row():
                    submit_btn = gr.Button("🚀 生成回复", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空历史")

            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="🤖 模型回复",
                    placeholder="调整滑动条后点击生成，查看风格插值效果...",
                    lines=10,
                    elem_classes="response-box",
                )
                history_display = gr.Markdown(
                    value="*（暂无对话历史）*",
                    elem_classes="history-box",
                )

        # Preset examples
        gr.Examples(
            examples=[
                ["我最近失恋了，每天都好难过", 0.8, 0.1, 0.3, 0.3],
                ["我想换工作但不知道怎么准备", 0.2, 0.8, 0.4, 0.2],
                ["考试失败了，觉得自己好没用", 0.7, 0.2, 0.9, 0.4],
                ["被人欺负了，心里很委屈", 0.9, 0.3, 0.6, 0.8],
                ["工作生活平衡不好，总感觉很累", 0.5, 0.6, 0.3, 0.5],
            ],
            inputs=[user_input, s1, s2, s3, s4],
            label="📋 预设场景 - 点击加载",
        )

        submit_btn.click(
            fn=on_generate,
            inputs=[user_input, s1, s2, s3, s4, temperature],
            outputs=[output, history_display],
        )
        clear_btn.click(fn=on_clear, outputs=[user_input, history_display])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-1.5B-Instruct", help="Base model name")
    parser.add_argument("--lora_dir", default="outputs/lora", help="Directory with style LoRA adapters")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    controller = StyleController(args.model_name, args.lora_dir, args.device)

    demo = create_demo(controller)
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
