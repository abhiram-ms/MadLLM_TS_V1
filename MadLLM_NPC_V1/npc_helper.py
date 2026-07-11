import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken
from pathlib import Path

tokenizer = tiktoken.get_encoding("gpt2")


vocab_size = tokenizer.n_vocab
num_transformer_blocks = 6
maxlen = 128
embed_dim = 192
num_heads = 6
feed_forward_dim = int(2/3 * 4 * embed_dim)
batch_size = 24
num_epochs = 3


class StoryDataset:
    def __init__(self,stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>', \
                        allowed_special={'<|endoftext|>'})[0]
        
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={'<|endoftext|>'})
        if len(tokens) > self.maxlen:
           tokens = tokens[:self.maxlen - 1]
           tokens.append(self.end_token)
            
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


def generate_text(model, start_tokens, max_new_tokens=50, temperature=1.0):
    tokens = list(start_tokens)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / temperature

        next_token = int(jnp.argmax(next_token_logits))

        if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)



def generate_story(model, story_prompt, temperature, max_new_tokens):
    start_tokens = tokenizer.encode(story_prompt)[:maxlen]
    generated = generate_text(model, start_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return generated



# --- MODEL DEFINITION ---

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        super().__init__()
        self.token_emb = nnx.Embed(vocab_size, embed_dim,rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim,rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)
    
#--------------Transformer Block ----------------------------------

class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim=None,
        dropout_rate=0.0,
        *,
        rngs
    ):
        # CPU friendly default: smaller FFN
        if ff_dim is None:
            ff_dim = embed_dim * 2

        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs
        )

        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)

        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.ff1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            rngs=rngs
        )

        self.ff2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            rngs=rngs
        )

        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, mask=None):
        # Pre-normalization attention block
        attn_input = self.ln1(x)
        attn_out = self.attention(attn_input, mask=mask)
        attn_out = self.dropout1(attn_out)

        x = x + attn_out

        # Feed-forward block
        ff_input = self.ln2(x)
        ff_out = self.ff1(ff_input)
        ff_out = jax.nn.gelu(ff_out, approximate=True)
        ff_out = self.ff2(ff_out)
        ff_out = self.dropout2(ff_out)

        x = x + ff_out

        return x
    
#------------------------------MadLLM Model Definition ----------------------------------
    
class MadLLM_NPC_V1(nnx.Module):

    def __init__(self, maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                 feed_forward_dim=feed_forward_dim, num_transformer_blocks=num_transformer_blocks, *, rngs=nnx.Rngs(0)):

        self.maxlen = maxlen

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)

        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
            ])

        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)
        
    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)

        return logits