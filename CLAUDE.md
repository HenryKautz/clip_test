# CLAUDE.md

## Project Overview

CLIP demo program that computes cosine similarity between an image and a text sentence using the `openai/clip-vit-large-patch14-336` model (the vision encoder used by LLaVA-1.5). Embeddings are 768-dimensional.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python clip_demo.py <image_file>
# then type a sentence on stdin
```

## Key Dependencies

- `torch` / `torchvision` — tensor operations
- `transformers` — Hugging Face CLIP model and processor
- `Pillow` — image loading
