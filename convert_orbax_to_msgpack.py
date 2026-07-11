from pathlib import Path
import json

import flax.nnx as nnx
from flax import serialization
from orbax import checkpoint

from inference_helper import MadLLM


# -------------------------
# Load config
# -------------------------
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


# -------------------------
# Recreate model
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
# Restore Orbax checkpoint
# -------------------------
checkpoint_path = (Path(__file__).parent / "small_checkpoint.orbax").resolve()
print("Loading Orbax checkpoint from:", checkpoint_path)

checkpointer = checkpoint.PyTreeCheckpointer()

state = nnx.state(model)

restored_state = checkpointer.restore(
    checkpoint_path.as_posix(),
    item=state
)

nnx.update(model, restored_state)

print("Orbax checkpoint restored successfully.")


# -------------------------
# Convert NNX State to pure dict
# -------------------------
final_state = nnx.state(model)

pure_state = nnx.to_pure_dict(final_state)


# -------------------------
# Save pure dict as msgpack
# -------------------------
output_path = Path("madllm_state.msgpack")

output_path.write_bytes(
    serialization.to_bytes(pure_state)
)

print("Saved msgpack checkpoint to:", output_path.resolve())