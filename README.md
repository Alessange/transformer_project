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


# last_gpt2.py
The code is a script for fine-tuning the GPT-2 model(Chinese base) on a custom dataset using the SophiaG optimizer. It trains the model for a specified number of epochs and saves the trained model.

The GPT-2 Fine-tuning with SophiaG Optimizer project provides a framework for fine-tuning the GPT-2 language model(Chinese base) on a custom dataset using the SophiaG optimizer. By leveraging the SophiaG optimizer's enhancements, this project aims to improve the training performance of GPT-2 models. It includes code for preparing the dataset, initializing the model and tokenizer, defining the training settings, and conducting the fine-tuning process. The resulting fine-tuned model can be used for various natural language processing tasks, such as text generation and language understanding. The project also provides a script for generating text using the trained model. With the flexibility of the MIT License, users are free to use, modify, and distribute the code according to their needs. This project builds upon the popular Transformers library by Hugging Face and acknowledges the contributions of the open-source community. It offers an accessible and customizable solution for those interested in exploring and improving GPT-2 models.

For more information on optimizer SophiaG, please see the [link](https://github.com/Liuhong99/Sophia)

# text_generate.py
The code snippet demonstrates how to use the fine-tuned GPT-2 model for text generation. The GPT-2 model and tokenizer are loaded from the specified paths. The TextGenerationPipeline is then created, utilizing the model and tokenizer, to generate text based on user input. The generated text is printed to the console.

To use the code, simply run the script and provide a keyword as input when prompted. The model will generate text based on the provided keyword.

This code provides a practical example of using a fine-tuned GPT-2 model for text generation tasks. It can be extended and customized for various applications, such as chatbots, creative writing, or content generation. Feel free to modify and adapt the code to suit your specific needs.

Please note that you need to replace the model path ""(/home/sxhuang/gpt2_new_last)"" with the actual path to your fine-tuned GPT-2 model.

# NBCE_decode.py
This script takes a starting sentence (prompt) and generates a story using the fine-tuned GPT-2 model.

The script starts by loading the fine-tuned GPT-2 model and its associated tokenizer.

It then requests an input prompt from the user. This prompt is encoded into a format suitable for the model. The encoded prompt is then passed to the model, which generates a sequence of tokens in response.

The encoding part is improved by a method named NBCE(Naive Bayes-based Context Extension), developed by 苏建林. More detail can be found [苏剑林. (May. 23, 2023). 《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度 》](https://kexue.fm/archives/9617)



