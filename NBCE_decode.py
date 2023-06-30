import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("./gpt2_story")

prompt = input("Enter a starting sentence for the story: ")
input_ids = tokenizer.encode(prompt, return_tensors='pt')
attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
model = model.to(device)

beta, eta = 0.25, 0.1
max_length = 1000  # maximum length
story = []

for _ in range(max_length):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1]
    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
    entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
    k = entropy.argmin() + 1
    logits_max = logits[k]
    logits_uncond = logits[0]
    logits_merged = (1 + beta) * logits_max - beta * logits_uncond
    logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
    next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
    if next_token[0] == tokenizer.eos_token_id:
        break
    story.append(tokenizer.decode(next_token[0]))
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

print(" ".join(story))
