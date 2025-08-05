from transformers import AutoTokenizer, LlamaForCausalLM

# Make sure you are logged in or pass the token explicitly
HF_TOKEN = "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"  # Replace with your token

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_auth_token=HF_TOKEN  # This ensures authentication
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=HF_TOKEN)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("\n🔹 AI Response:", response)
