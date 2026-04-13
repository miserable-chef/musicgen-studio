# MusicGen Studio

> Text-to-music generation using [Meta's MusicGen](https://arxiv.org/abs/2306.05284) — Streamlit UI with prompt examples, model selection, duration control, and WAV download.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![MusicGen](https://img.shields.io/badge/Meta-MusicGen-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

Type a description of music → MusicGen composes it → play in browser or download as WAV.

## Quick Start

```bash
git clone https://github.com/miserable-chef/musicgen-studio
cd musicgen-studio
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads model weights (~300MB for small, ~1.5GB for medium).

## Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `musicgen-small` | ~300MB | Fast | Good |
| `musicgen-medium` | ~1.5GB | Moderate | Better |

## Prompt Tips

Be specific about:
- **Genre/style** — "lo-fi hip hop", "Indian classical", "80s synth-pop"
- **Instruments** — "sitar and tabla", "acoustic guitar", "synthesizer arpeggios"
- **Mood** — "melancholic", "upbeat", "meditative", "epic"
- **Tempo** — "slow and calm", "fast-paced", "steady groove"

## Tech Stack

- [Meta MusicGen](https://github.com/facebookresearch/audiocraft) via HuggingFace Transformers
- Streamlit for the UI
- PyTorch for inference (CPU/CUDA)
- SciPy for WAV encoding
