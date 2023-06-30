import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")

with open("/home/sxhuang/transformer_project-main/output.json", "r") as file:
    stories = json.load(file)

texts = [story["content"] for story in stories]


def load_dataset(texts, tokenizer):
    tokenized_texts = tokenizer(texts, truncation=True, padding=False,
                                return_tensors='pt')
    return TextDataset(tokenizer=tokenizer,
                       examples=tokenized_texts["input_ids"], block_size=128)


def load_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


dataset = load_dataset(texts, tokenizer)
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
