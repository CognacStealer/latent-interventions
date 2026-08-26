import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Starting the program")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             dtype=torch.float16, 
                                             device_map="auto")
print("The model is loaded successfully")
prompt = "The movie was amazing"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states  = output.hidden_states
print(hidden_states)