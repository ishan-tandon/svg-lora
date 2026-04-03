from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
).to("cuda")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "./qwen-svg-lora/lora_adapter")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-3B-Instruct")

prompt = """<|im_start|>system
You are an expert SVG coder. Return ONLY raw SVG code.<|im_end|>
<|im_start|>user
Generate SVG for: A simple red circle on a white background.<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== OUTPUT ===")
print(result)
