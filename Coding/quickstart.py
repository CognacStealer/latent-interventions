from llm_steer import Steer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── Load model ────────────────────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── Apply steering ─────────────────────────────────────────────────────────────
steered = Steer(model, tokenizer)
steered.add(layer_idx=14, coeff=0.3, text="cheerful enthusiastic positive")

# ── Extract latent vectors via hooks ──────────────────────────────────────────
latent_vectors = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # output can be a tuple (hidden_state, ...) or just a tensor — handle both
        hidden = output[0] if isinstance(output, tuple) else output

        # Some layers return (seq_len, hidden_dim) without batch dim — unsqueeze
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)   # → (1, seq_len, hidden_dim)

        latent_vectors[layer_idx] = hidden.detach().clone()
        print(f"  Hook fired layer {layer_idx}: shape={tuple(hidden.shape)}")
    return hook_fn

TARGET_LAYERS = [0, 7, 14, 20, 27]

hooks = []
for idx in TARGET_LAYERS:
    layer = model.model.layers[idx]
    h = layer.register_forward_hook(make_hook(idx))
    hooks.append(h)

# ── Generate ───────────────────────────────────────────────────────────────────
prompt = "Explain how solar panels work"
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

for h in hooks:
    h.remove()

# ── Decode ─────────────────────────────────────────────────────────────────────
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print("\n=== Response ===")
print(response)

# ── Inspect latent vectors ─────────────────────────────────────────────────────
print("\n=== Latent Vectors (last token at each layer) ===")
for layer_idx, tensor in sorted(latent_vectors.items()):
    # tensor is now guaranteed to be (batch, seq_len, hidden_dim)
    last_token_vec = tensor[0, -1, :]
    print(f"Layer {layer_idx:>2}: shape={tuple(tensor.shape)}  "
          f"mean={last_token_vec.mean().item():.4f}  "
          f"std={last_token_vec.std().item():.4f}  "
          f"norm={last_token_vec.norm().item():.4f}")

# ── Steering vector via llm_steer internal ────────────────────────────────────
print("\n=== Steering Vector ===")
steering_text = "cheerful enthusiastic positive"
raw_tensor = steered._capture_tensor(
    layer_idx=14,
    tokens=torch.tensor(
        tokenizer.encode(steering_text, add_special_tokens=False)
    ).unsqueeze(0)
)
# raw_tensor may also be 2D here — handle it
if raw_tensor.dim() == 2:
    raw_tensor = raw_tensor.unsqueeze(0)

steering_vec = raw_tensor[0].mean(dim=0)
print(f"Shape : {tuple(steering_vec.shape)}")
print(f"Norm  : {steering_vec.norm().item():.4f}")
print(f"Mean  : {steering_vec.mean().item():.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
torch.save({
    "latent_vectors": latent_vectors,
    "steering_vector": steering_vec,
    "prompt": prompt,
}, "latent_vectors.pt")
print("\nSaved to latent_vectors.pt")