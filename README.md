# transformer_project
The Transformer and NLP model has been employed for text classification and prediction in this project. Our primary objective is the implementation of the NLP model using data extracted from various websites.


# Crawler
This Python script is a multi-threaded web scraper that extracts and processes story data from the website 'www.xigushi.com'. The specific story data it extracts include categories, titles, content, and word counts. The story data span across various categories such as humor, children, love, career, inspiration, philosophy, campus, life, fable, famous people, family love, and friendship.

The script accomplishes the following tasks:

- Extracts links for different categories from the HTML content.
- Scrapes the index pages for each category, retrieving the URLs for the individual story pages.
- Extracts the story data from each story page.
- Parses and cleans the extracted data.
- Appends the cleaned data into a list.
- Writes the resulting data into a JSON file.
The script uses concurrent futures (ThreadPoolExecutor) for efficient parallel processing, thereby improving its speed.


# BERT_model(category)
This Python script trains a text classification model using the Transformer library and a dataset of categorized stories.

Here's a detailed description of what the script does:

- Dataset class: This class extends the PyTorch torch.utils.data.Dataset class. It reads a JSON file containing the story data using Hugging Face's datasets library and encodes the category labels as integers using sklearn's LabelEncoder.

- Data loading and preprocessing: The script tokenizes the content using the Hugging Face's Transformer library's BERT tokenizer (Chinese variant). It also packs the story data into a DataLoader that provides batches of stories.

- Model definition: The script defines a neural network model that uses a pre-trained BERT model (Chinese variant) from Hugging Face's Transformer library. This model serves as a feature extractor. The final layer of this model is a fully connected (Linear) layer with an output size equivalent to the number of unique categories in the dataset, which classifies the stories into different categories.

- Training: Finally, the script trains this model using the AdamW optimizer and the cross-entropy loss function. It loops over the DataLoader, passes the data through the model, computes the loss, and backpropagates the gradients to update the model's weights. Every 15 steps, it computes the training accuracy and prints it along with the loss.

To save the model locally after the training loop, use the following code:

_If you wish to save the model locally, try the code after the training loop_

```ruby
torch.save(model.state_dict(), "/path/to/save/bert_model_category.pth")
```

During the training loop, the gradients of the pre-trained BERT model's parameters are not updated (they are "frozen"). Only the weights of the final Linear layer are learned from scratch. This is a common approach in transfer learning, where we use a model pre-trained on a large dataset (in this case, BERT) and fine-tune it on a smaller, specific task (here, the story classification task).


# gpt2_text_model.py
This script fine-tunes a GPT-2 model with custom data, which should be in the form of a list of text stories. These stories are read into the script, tokenized, and used to create a TextDataset object. The data collator function prepares samples for the model.

The TrainingArguments object sets the training configuration parameters like the output directory, number of epochs, batch size, and save steps.

A Trainer object is created by passing the model, training arguments, data collator, and dataset to it. This Trainer object trains the model.

Once the training is complete, the script saves the model. This fine-tuned model is then ready for use in generating new stories.


# NBCE_decode.py
This script takes a starting sentence (prompt) and generates a story using the fine-tuned GPT-2 model.

The script starts by loading the fine-tuned GPT-2 model and its associated tokenizer.

It then requests an input prompt from the user. This prompt is encoded into a format suitable for the model. The encoded prompt is then passed to the model, which generates a sequence of tokens in response.

The encoding part is improved by a method named NBCE(Naive Bayes-based Context Extension), developed by 苏建林. More detail can be found [苏剑林. (May. 23, 2023). 《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度 》](https://kexue.fm/archives/9617)

