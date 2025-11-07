import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.amp import autocast
from transformers import BitsAndBytesConfig

model_path = "./quantized_model"

# Load the 8-bit quantized model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the pad_token_id explicitly (if not already set)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Fallback: Set pad token to eos token if None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Free up memory before running inference (useful if you're working with a large model)
torch.cuda.empty_cache()

# Use mixed precision with updated `autocast` (corrected to use `torch.amp.autocast('cuda')`)
with autocast("cuda"):  # Use mixed precision if on CUDA
    while True:
        user_input = input("Input your prompt: ")
        formatted_input = f"issue: {user_input} solution:"
        
        # Tokenize and generate response, ensure attention_mask is included
        encoding = tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Get the input_ids and attention_mask from the encoding
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)  # Ensure attention_mask is passed

        # Free memory before each inference step to prevent overflow
        torch.cuda.empty_cache()

        try:
            # Run inference with mixed precision to reduce memory usage
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=250, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory! Freeing up some space.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        # Decode the output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"Model's Response: {output_text}")
        
        # Exit condition (optional)
        if user_input.lower() == 'exit':
            break