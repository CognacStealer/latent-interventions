from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")

text = "I hate everything about this"
tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)

# Your hidden state at layer 6
activation = cache["resid_post", 6]
print(activation.shape)  # should be [1, seq_len, 768]