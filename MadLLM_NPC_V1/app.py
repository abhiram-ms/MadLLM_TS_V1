#app.py

import json
from pathlib import Path

#disable gradio's SSR mode to avoid issues with JAX/Flax state management in the UI
import os
os.environ["GRADIO_SSR_MODE"] = "False"

# Suppress harmless asyncio cleanup noise on Hugging Face Spaces.
try:
    import asyncio.base_events as base_events

    original_del = getattr(base_events.BaseEventLoop, "__del__", None)

    if original_del is not None:
        def patched_del(self):
            try:
                original_del(self)
            except ValueError as e:
                if "Invalid file descriptor" not in str(e):
                    raise

        base_events.BaseEventLoop.__del__ = patched_del
except Exception:
    pass

import gradio as gr
import flax.nnx as nnx
#from orbax import checkpoint
from flax import serialization

from MadLLM_NPC_V1.npc_helper import MadLLM, generate_story


# -------------------------
# Load model config
# -------------------------
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


# -------------------------
# Recreate model architecture
# -------------------------
model = MadLLM(
    maxlen=config["maxlen"],
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    feed_forward_dim=config["feed_forward_dim"],
    num_transformer_blocks=config["num_transformer_blocks"],
    rngs=nnx.Rngs(0)
)


# -------------------------
# Load msgpack checkpoint
# -------------------------
checkpoint_path = (Path(__file__).parent / "madllm_state.msgpack").resolve()

print("Loading model state from:", checkpoint_path)

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"madllm_state.msgpack not found at: {checkpoint_path}"
    )

state = nnx.state(model)

pure_state_template = nnx.to_pure_dict(state)

restored_pure_state = serialization.from_bytes(
    pure_state_template,
    checkpoint_path.read_bytes()
)

restored_pure_state = nnx.restore_int_paths(restored_pure_state)

# IMPORTANT:
# This modifies `state` in-place and returns None.
nnx.replace_by_pure_dict(state, restored_pure_state)

# Now update model using the modified state.
nnx.update(model, state)

print("Model loaded successfully from msgpack.")


# -------------------------
# Gradio inference function
# -------------------------
def create_story(story_prompt, temperature, max_tokens):
    if not story_prompt or not story_prompt.strip():
        return "Please enter a story prompt."

    try:
        return generate_story(
            model=model,
            story_prompt=story_prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_tokens)
        )
    except Exception as e:
        return f"Error while generating story: {str(e)}"


# -------------------------
# Gradio UI
# -------------------------
demo = gr.Interface(
    fn=create_story,
    inputs=[
        gr.Textbox(
            label="Story Prompt",
            placeholder="Once upon a time, a small frog went to the forest..."
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.8,
            step=0.05,
            label="Temperature"
        ),
        gr.Slider(
            minimum=10,
            maximum=200,
            value=50,
            step=1,
            label="Max Tokens"
        )
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="MadLLM_TS_V1",
    description="A tiny from-scratch JAX/Flax language model trained on TinyStories and NFT datasets for npc."
)

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False
)