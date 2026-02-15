#!/usr/bin/env python3
"""
Demonstrate CLIP by computing cosine similarity between an image and a text sentence.

Uses openai/clip-vit-large-patch14-336, the vision encoder used by LLaVA-1.5.

Usage:
    python clip_demo.py image.jpg
    (then type a sentence and press Enter)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14-336"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_file>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}", file=sys.stderr)
        sys.exit(1)

    print("Enter a sentence:", file=sys.stderr)
    sentence = input().strip()
    if not sentence:
        print("Error: empty sentence", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model {MODEL_NAME}...", file=sys.stderr)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = processor(text=[sentence], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds  # (1, dim)
        text_emb = outputs.text_embeds    # (1, dim)

    cosine = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()

    print(f"Cosine similarity: {cosine:.4f}")


if __name__ == "__main__":
    main()
