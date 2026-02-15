# CLIP Demo

Demonstrates OpenAI's CLIP model by computing the cosine similarity between an image and text sentences. Uses `openai/clip-vit-large-patch14-336`, the vision encoder used by LLaVA-1.5.

## Setup

```bash
pip install -r requirements.txt
```

The model weights (~1.7 GB) are downloaded and cached automatically on first run.

## Usage

```bash
python clip_demo.py
```

This launches a Gradio web UI in your browser. Upload an image, enter up to 4 sentences, and click the **CLIP** button to compute cosine similarities.

Values closer to 1.0 indicate the image and text are semantically similar; values near 0.0 indicate they are unrelated. Typical scores for good matches are in the 0.20–0.35 range.
