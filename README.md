# WEAT Analysis of W2V & NLP algorithms
- Submission for Mathwork's Responsible AI Hackathon 2023: https://northeastern-mw-hackathon.devpost.com/
    - Won Most Cross-Disciplinary Award  
    - Placed 4th overall
- Video link: https://youtu.be/5I6dWDDyJjA

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
**Detailed in our story here** https://drive.google.com/drive/u/0/folders/1orZPQWuP_4145IivD9gth3GF58VjbR4S 

**Screenshots of the results can be found in the results folder**

## How to run the program
1. Create a new virtual environment called Hackathon with  ```python3 -m venv Hackathon```
2. Activate the virtual environment in the working directory with ```.\Hackathon\Scripts\activate``` (windows)
3. Install the requirements with  ```pip install -r requirements.txt```

### Method 1: Use the word-embeddings in the csv files
1. run  ```python WEAT2.py``` to run your own WEAT test on the biased, unbiased, biased + unbiased, and google-news word embeddings

#### Generate own word-embeddings based on our provided code
1. run  ```python training_model.py``` to generate your own data based on the training model
	1. EXTRA: modify the google news 300 pre-trained model to a pre-trained vector of your choosing such as a GloVe or another pre-trained W2V model
		1. Go to pre-trained.py and modify ```model = api.load('word2vec-google-news-300')``` to ```model = api.load(YOURMODELHERE)```
2. run ```python WEAT2.py``` to analyze that data

### Method 2: Create, train, and use your own model's word embeddings
1. go to training_models.py and find where it has ```biased_file = 'Training_biased unbiased_file = 'Training_unbiased'```
	1. Edit Training_biased.txt or Training.unbiased.txt with your own [Project Gutenberg](https://www.gutenberg.org/) books. Make sure to extract their book ids from the Book ID fiedl
	2. Or change the path to your own provided corpus of texts and follow the example of word2vecTest.py under the Old_Files folder, setting the corpus to your own sample of text and iterating through each word of the txt file
	3. EXTRA: modify the google news 300 pre-trained model to a pre-trained vector of your choosing such as a GloVe or another pre-trained W2V model
		1. Go to pre-trained.py and modify ```model = api.load('word2vec-google-news-300')``` to ```model = api.load(YOURMODELHERE)```
2. Change the export names of the csv file to whatever you choose
3. run ```python training_model.py```
	1. This should export the biased file and unbiased files as csv files with the name you gave it
4. change the path for the ```filenames = ["Data/biased_trainedmodel_word_vectors.csv", "Data/unbiased_trainedmodel_word_vectors.csv", "Data/all_trainedmodel_word_vectors.csv", "Data\GoogleNews_word_vectors.csv"]``` to the path and name of the csv files you chose
5. run ```python WEAT2.py```
    
