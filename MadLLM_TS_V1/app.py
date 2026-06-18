import json
from pathlib import Path

import gradio as gr
import flax.nnx as nnx
from orbax import checkpoint

from inference_helper import MadLLM, generate_story


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
# Load Orbax checkpoint
# -------------------------
checkpoint_path = (Path(__file__).parent / "small_checkpoint.orbax").resolve()

print("Loading checkpoint from:", checkpoint_path)

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at: {checkpoint_path}"
    )

checkpointer = checkpoint.PyTreeCheckpointer()

state = nnx.state(model)

restored_state = checkpointer.restore(
    checkpoint_path.as_posix(),
    item=state
)

nnx.update(model, restored_state)


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

demo.launch()