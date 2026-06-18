from datasets import load_dataset
from itertools import islice

OUTPUT_FILE = "TinyStories-10000.txt"
NUM_STORIES = 10000

# Streaming avoids downloading the full 7GB+ dataset
dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, row in enumerate(islice(dataset, NUM_STORIES)):
        story = row["text"].strip()
        f.write(story)
        f.write("\n<|endoftext|>\n")

print(f"Saved {NUM_STORIES} stories to {OUTPUT_FILE}")