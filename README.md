# CLIP Demo

Demonstrates OpenAI's CLIP model by computing the cosine similarity between an image and a text sentence. Uses `openai/clip-vit-large-patch14-336`, the vision encoder used by LLaVA-1.5.

## Setup

```bash
pip install -r requirements.txt
```

The model weights (~1.7 GB) are downloaded and cached automatically on first run.

## Usage

```bash
python clip_demo.py <image_file>
```

The program reads a sentence from stdin, computes CLIP embeddings for both the image and the sentence, and prints their cosine similarity.

### Example

```bash
python clip_demo.py photo.jpg
```

```
Enter a sentence:
a dog playing in the park
Cosine similarity: 0.2834
```

Values closer to 1.0 indicate the image and text are semantically similar; values near 0.0 indicate they are unrelated.
