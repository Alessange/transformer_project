import torch
from transformers import AutoModelForCausalLM, BertTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load your model
model_path = '/home/sxhuang/gpt2_new_last'
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)  # Ensure the model is on the right device
model.eval()  # Set model to evaluation mode for inference

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

prompt = input("Enter a starting sentence for the story: ")
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
attention_mask = torch.ones_like(input_ids).to(device)

beta, eta = 0.25, 0.1
max_length = int(input('输入所需长度:'))  # maximum length
story = []

for _ in range(max_length):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1]
    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
    entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
    k = min(entropy.argmin().item(), logits.shape[0]-1)
    logits_max = logits[k]
    logits_uncond = logits[0]
    logits_merged = (1 + beta) * logits_max - beta * logits_uncond
    logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
    next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
    if next_token[0] == tokenizer.eos_token_id:
        break
    decode_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
    if decode_text != '[ P A D ]':
        story.append(decode_text)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

generated_text = " ".join(story)
print(generated_text)
