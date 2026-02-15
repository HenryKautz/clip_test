#!/usr/bin/env python3
"""
Demonstrate CLIP by computing cosine similarity between an image and text sentences.

Uses openai/clip-vit-large-patch14-336, the vision encoder used by LLaVA-1.5.
Provides a Gradio web UI with image upload and up to 4 sentence inputs.

Usage:
    python clip_demo.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14-336"

print(f"Loading model {MODEL_NAME}...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")


def compute_similarities(image, s1, s2, s3, s4):
    if image is None:
        return "", "", "", ""

    sentences = [s1, s2, s3, s4]
    results = []

    # Collect non-empty sentences and their indices
    active = [(i, s.strip()) for i, s in enumerate(sentences) if s and s.strip()]

    if not active:
        return "", "", "", ""

    texts = [s for _, s in active]
    image = Image.fromarray(image).convert("RGB")

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds   # (1, dim)
        text_embs = outputs.text_embeds    # (N, dim)

    cosines = torch.nn.functional.cosine_similarity(image_emb, text_embs)

    # Map results back to the 4 slots
    result_map = {}
    for (idx, _), cos in zip(active, cosines):
        result_map[idx] = f"{cos.item():.4f}"

    return tuple(result_map.get(i, "") for i in range(4))


with gr.Blocks(title="CLIP Demo") as demo:
    gr.Markdown("# CLIP Demo\n`openai/clip-vit-large-patch14-336` — the vision encoder used by LLaVA-1.5")

    with gr.Row():
        image_input = gr.Image(label="Image", type="numpy", height=300)

        with gr.Column():
            text_inputs = []
            score_outputs = []
            for i in range(4):
                with gr.Row():
                    t = gr.Textbox(label=f"Sentence {i + 1}", scale=4)
                    s = gr.Textbox(label="Cosine", scale=1, interactive=False)
                    text_inputs.append(t)
                    score_outputs.append(s)

            with gr.Row():
                clip_btn = gr.Button("CLIP", variant="primary")
                exit_btn = gr.Button("Exit", variant="stop")

    clip_btn.click(
        fn=compute_similarities,
        inputs=[image_input] + text_inputs,
        outputs=score_outputs,
    )
    exit_btn.click(fn=lambda: os._exit(0))

demo.launch(inbrowser=True)
