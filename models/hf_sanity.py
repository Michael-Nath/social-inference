import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# BOS token, then "Hello, my name is"
# input_ids = torch.tensor([[128000, 13347, 856, 836, 374, 8388]])
# Encode the text "I like to" using the tokenizer
input_ids = torch.tensor([tokenizer.encode("I like to")])
print(input_ids)
# position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
position_ids = torch.arange(len(input_ids)).unsqueeze(0)

# Get model outputs
with torch.no_grad():
    outputs = model(input_ids=input_ids, position_ids=position_ids)

# Get the logits from the output
logits = outputs.logits

# Print the logits
next_token = logits[0][-1].argmax()
print(tokenizer.decode(next_token))