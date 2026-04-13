"""
MusicGen Studio — Text-to-music generation using Meta's MusicGen.
Enter a text prompt, pick a model size and duration, generate and download.
"""

import io
import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
import torch

st.set_page_config(
    page_title="MusicGen Studio",
    page_icon="🎵",
    layout="centered",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 MusicGen Studio")
    st.caption("Powered by Meta's MusicGen via HuggingFace Transformers")
    st.divider()

    model_size = st.selectbox(
        "Model",
        ["small", "medium"],
        help="Small (~300MB, fast) · Medium (~1.5GB, better quality)",
    )
    duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10, step=5)
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="Higher = more faithful to prompt, lower = more creative",
    )
    st.divider()
    st.markdown(
        "**Model:** `facebook/musicgen-{}` \n\n"
        "**Device:** `{}` \n\n"
        "**Docs:** [MusicGen Paper](https://arxiv.org/abs/2306.05284)".format(
            model_size,
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )


# ── model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MusicGen model — first run downloads weights...")
def load_model(size: str):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    model_id = f"facebook/musicgen-{size}"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return processor, model


# ── prompt examples ───────────────────────────────────────────────────────────
EXAMPLES = [
    "Upbeat electronic dance music with a punchy bassline and synthesizer arpeggios",
    "Calm acoustic guitar fingerpicking, warm and melancholic, suitable for studying",
    "Epic cinematic orchestral score with brass swells and dramatic timpani",
    "Lo-fi hip hop beat with jazzy chords, vinyl crackle, and mellow vibes",
    "Indian classical raga with sitar and tabla, meditative and hypnotic",
    "80s synth-pop with catchy melody, drum machine, and retro feel",
    "Intense metal guitar riff with double bass drum and distortion",
    "Ambient soundscape with nature sounds, soft piano, and gentle pads",
]


# ── main UI ───────────────────────────────────────────────────────────────────
st.title("MusicGen Studio")
st.caption("Describe the music you want — MusicGen will compose it.")

st.markdown("**Prompt examples — click to use:**")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example[:55] + "…", key=f"ex{i}"):
        st.session_state["prompt"] = example

prompt = st.text_area(
    "Your prompt",
    value=st.session_state.get("prompt", ""),
    placeholder="e.g. Calm lo-fi beats with jazzy piano chords and soft rain in the background",
    height=90,
)

generate = st.button("🎵 Generate", type="primary", disabled=not prompt.strip())

if generate and prompt.strip():
    with st.spinner(f"Composing {duration}s of music with musicgen-{model_size}…"):
        try:
            processor, model = load_model(model_size)

            # tokens_per_second ≈ 50 for MusicGen
            max_new_tokens = int(duration * 50)

            inputs = processor(
                text=[prompt.strip()],
                padding=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=guidance_scale,
                    do_sample=True,
                )

            sampling_rate = model.config.audio_encoder.sampling_rate
            audio_np = audio_values[0, 0].cpu().numpy()

            # normalise to int16 for WAV
            audio_int16 = (audio_np / np.abs(audio_np).max() * 32767).astype(np.int16)

            # write to in-memory WAV buffer
            buf = io.BytesIO()
            wav.write(buf, sampling_rate, audio_int16)
            buf.seek(0)

            st.success(f"Generated {duration}s · {sampling_rate/1000:.0f}kHz · musicgen-{model_size}")
            st.audio(buf, format="audio/wav")

            st.download_button(
                label="⬇ Download WAV",
                data=buf.getvalue(),
                file_name=f"musicgen_{model_size}_{duration}s.wav",
                mime="audio/wav",
            )

            with st.expander("Generation details"):
                st.json({
                    "prompt": prompt.strip(),
                    "model": f"facebook/musicgen-{model_size}",
                    "duration_seconds": duration,
                    "sampling_rate": sampling_rate,
                    "guidance_scale": guidance_scale,
                    "samples_generated": len(audio_np),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                })

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.info("If this is a memory error, try switching to the **small** model in the sidebar.")
