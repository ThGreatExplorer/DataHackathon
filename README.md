# WEAT Analysis of W2V & NLP algorithms
- Submission for Mathwork's Responsible AI Hackathon 2023: https://northeastern-mw-hackathon.devpost.com/
    - Won Most Cross-Disciplinary Award  
    - Placed 4th overall

## Team
 - Daniel Yu
 - James Lewis
 - Spencer Chin
 - Ajith George

## Problem Statement
Inspired by human biases in translation, specifically how translator's bias could affect the meaning and interpretation of books and the original author's intention, we decided to create our own project on the idea of detecting and quantifying bias in human and machine translated tasks.

Note:
Project was heavily influenced by: https://developers.googleblog.com/2018/04/text-embedding-models-contain-bias.html

## Approach
1. **Dataset**
	1. Used Project Gutenberg library to find and curate a dataset of books 
		1. Used projectgutenbergpy python library
	2. Rigorously classified books as biased or unbiased 
	3. Split data into training_biased, training_unbiased, validation_bias, and validation_unbiased
2. **Trained W2V models on dataset**
	1. used gensim library to train our own W2V models based on training_biased, training_unbiased, and both training_biased and training_unbiased
		1. extracted word embeddings
	2. imported pre-trained google-news-300 vector
		1. extracted word embeddings
3. **Ran WEAT analysis on W2V models**
	1. compared our trained W2V models to the pre-trained google-news-300 vector
		1. Randomly sampled 2000 target words from each model against attribute word pairs of each word in negative_words.txt and positive_words.txt as our standard set
4. **Interpreted WEAT metric analysis results**
	1. compared WEAT results of biased vs unbiased vs biased and unbiased

Screenshots of the results can be found in the results folder

## How to run the program
1. Create a new virtual environment called Hackathon with  ```python3 -m venv Hackathon```
2. Activate the virtual environment in the working directory with ```.\Hackathon\Scripts\activate``` (windows)
3. Install the requirements with  ```pip install -r requirements.txt```
4. ...
