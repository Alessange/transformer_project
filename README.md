# transformer_project
applied Transformer and NLP model for text_classification and prediction


The main goal of this project is for NLP model implementation with the data get from the websites.

# Crawler

This Python script is a multi-threaded web scraper that extracts and processes story data from the website 'www.xigushi.com'. The specific story data it extracts includes categories, titles, content, and word counts. The story data are distributed across various categories such as humor, children, love, career, inspiration, philosophy, campus, life, fable, famous people, family love, and friendship.

The script performs the following tasks:

Extracts links for different categories from the HTML content.
Scrapes the index pages for each category, retrieving the URLs for the individual story pages.
Extracts the story data from each story page.
Parses and cleans the extracted data.
Appends the cleaned data into a list.
Writes the resulting data into a JSON file.
The use of concurrent futures (ThreadPoolExecutor) allows for efficient parallel processing, improving the speed of the script.

# BERT_model(category)

This Python script trains a text classification model using the Transformer library and a dataset of categorized stories.

Here's what the script does in detail:

Dataset class: This class extends the PyTorch torch.utils.data.Dataset class. It reads a JSON file containing the story data using the Hugging Face's datasets library and encodes the category labels as integers using sklearn's LabelEncoder.

Data loading and preprocessing: The script then tokenizes the content using the Hugging Face's Transformer library's BERT tokenizer (Chinese variant). It also packs the story data into a DataLoader that provides batches of stories.

Model definition: The script defines a neural network model that uses a pre-trained BERT model (again, Chinese variant) from Hugging Face's Transformer library, which is used as a feature extractor. The final layer of this model is a fully connected (Linear) layer with an output size equivalent to the number of unique categories in the dataset, which is used to classify the stories into different categories.

Training: Finally, the script trains this model using the AdamW optimizer and the cross-entropy loss function. It loops over the DataLoader, passes the data through the model, computes the loss, and backpropagates the gradients to update the model's weights. It also computes the training accuracy every 15 steps and prints it out along with the loss.

Note: In the training loop, the gradients of the pre-trained BERT model's parameters are not updated (they are "frozen") - only the weights of the final Linear layer are learned from scratch. This is a common approach when doing transfer learning, where we use a model pre-trained on a large dataset (in this case, BERT) and fine-tune it on a smaller, specific task (here, the story classification task).
