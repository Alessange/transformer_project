import torch
from transformers import TrainingArguments, Trainer, get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from sophia import SophiaG

tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")


class Datasets(torch.utils.data.Dataset):
    def __init__(self):
        with open(
                '/home/sxhuang/transformer_project-main/data/output.txt') as f:
            lines = f.readlines()
        contents = [line.strip() for line in lines]

        self.contents = contents

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        text = self.contents[item]
        encoding = tokenizer.encode_plus(text, truncation=True, max_length=512, padding="max_length", return_tensors='pt')
        encoding["labels"] = encoding["input_ids"].detach().clone()
        return encoding


dataset = Datasets()

model = AutoModelForCausalLM.from_pretrained('uer/gpt2-chinese-cluecorpussmall')

for param in model.parameters():
    param.requires_grad_(False)

for param in model.transformer.h[
    -5].parameters():  # Unfreezing the last transformer block
    param.requires_grad = True


def train():
    training_args = TrainingArguments(
        output_dir="./results",  # The output directory
        overwrite_output_dir=True,
        # overwrite the content of the output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=8,  # batch size for training
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
    )

    class SophiaGTrainer(Trainer):
        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.optimizer = SophiaG(self.model.parameters(), lr=2e-4,
                                     betas=(0.965, 0.99), rho=0.01,
                                     weight_decay=1e-1)
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = SophiaGTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model('gpt2_new_last')


# Run the training function
train()
