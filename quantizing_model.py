from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model with 4-bit quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
   quantization_config=quantization_config,  # Pass the quantization config here
    device_map="auto"  # Automatically place model on available device(s)
)

# Save the quantized model to disk
model_save_path = "./quantized_model"
model.save_pretrained(model_save_path)

# Save the tokenizer to disk as well
tokenizer.save_pretrained(model_save_path)

# Verify if model and tokenizer are saved correctly
print(f"Model and tokenizer saved at: {model_save_path}")