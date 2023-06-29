# transformer_project
applied Transformer and NLP model for text_classification and prediction


The main goal of this project is for NLP model implementation with the data get from the websites.

#Crawler

This Python script is a multi-threaded web scraper that extracts and processes story data from the website 'www.xigushi.com'. The specific story data it extracts includes categories, titles, content, and word counts. The story data are distributed across various categories such as humor, children, love, career, inspiration, philosophy, campus, life, fable, famous people, family love, and friendship.

The script performs the following tasks:

Extracts links for different categories from the HTML content.
Scrapes the index pages for each category, retrieving the URLs for the individual story pages.
Extracts the story data from each story page.
Parses and cleans the extracted data.
Appends the cleaned data into a list.
Writes the resulting data into a JSON file.
The use of concurrent futures (ThreadPoolExecutor) allows for efficient parallel processing, improving the speed of the script.

