import torch
from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("json",
                                    data_files='/Users/huangshixun/Desktop'
                                               '/Transformer/output.json',
                                    split=split)
        self.label_encoder = LabelEncoder().fit(self.dataset['category'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        content = self.dataset[i]['content']
        category = self.label_encoder.transform([self.dataset[i]['category']])[
            0]
        return content, category


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = Dataset('train')


def collate_fn(data):
    contents, categories = zip(*data)

    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=contents,
                                       truncation=True, padding='max_length',
                                       max_length=300, return_tensors='pt')
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    categories = torch.LongTensor(categories)

    return input_ids, attention_mask, token_type_ids, categories


loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=30,
                                     collate_fn=collate_fn, shuffle=True,
                                     drop_last=True)

pretrained = BertModel.from_pretrained('bert-base-chinese')

for param in pretrained.parameters():
    param.requires_grad_(False)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, len(set(dataset.dataset['category'])))

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        return out


model = Model()

optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids, categories) in enumerate(
        loader): # Move all tensors to GPU if available
    out = model(input_ids, attention_mask, token_type_ids)

    loss = criterion(out, categories)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 15 == 0:
        out = out.argmax(dim=-1)
        accuracy = (out == categories).sum().item() / len(categories)
        print(f"Step: {i}, Loss: {loss.item()}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "/Users/huangshixun/Desktop/Transformer/bert_model_category.pth")
