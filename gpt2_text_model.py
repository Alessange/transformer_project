import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last transformer layer for fine-tuning
for param in model.transformer.h[-3].parameters():
    param.requires_grad = True

with open("/Users/huangshixun/Desktop/Transformer/output.json", "r") as file:
    stories = json.load(file)

texts = [story["content"] for story in stories]


# Create a function to encode the texts
def encode(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


# Create a Dataset object
data = Dataset.from_dict({"text": texts})
dataset = data.map(encode, batched=True)


def load_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


data_collator = load_data_collator(tokenizer)

training_args = TrainingArguments(
    output_dir="./gpt2_story",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model()
