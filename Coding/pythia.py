from transformer_lens import HookedTransformer
import torch 
model = HookedTransformer.from_pretrained("pythia-70m")
prompt = "The weather today is sunny"
tokens = model.to_tokens(prompt)

logits , cache = model.run_with_cache(tokens)

print(model.to_str_tokens(prompt))
print(cache["blocks.4.hook_resid_post"].shape)

  # Print the first 10 values of the last token's residual stream from block 4


d_model = model.cfg.d_model
random_direction = torch.randn(d_model)

def steering_hook(resid,hook):
    return resid + 6.0 * random_direction


baseline_output = model.generate(prompt,max_new_tokens=20)
steered_output = model.generate(
    prompt, max_new_tokens=20,
    fwd_hooks=[("blocks.4.hook_resid_post", steering_hook)]
)

print("Baseline output:", baseline_output)
print("Steered output:", steered_output)