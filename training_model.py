import os
import re
import gzip
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gutenbergpy.textget
import csv
import pandas as pd
import numpy as np

# downloads stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')

# We want to program the word2vec to extract cosine similarities between words
# model trained on entire corpus of text 

biased_file = 'Training_biased'
unbiased_file = 'Training_unbiased'

# Read in the book IDs from the biased and unbiased files
with open(biased_file, 'r') as f:
    biased_ids = re.findall('\d+', f.read())

with open(unbiased_file, 'r') as f:
    unbiased_ids = re.findall('\d+', f.read())

book_data = []

for book_id in biased_ids:
    #try:
        # Process and Tokenize the text
        raw_book = gutenbergpy.textget.get_text_by_id(int(book_id))
        clean_book = gutenbergpy.textget.strip_headers(raw_book)
        processed_book = clean_book.decode('utf-8').replace("\n", " ").replace("\r", " ").replace(".", " ").replace(",", " ")
        tokens = word_tokenize(processed_book)

        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

        # Lemmatize the words
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # append list of words together into a single book's words
        book_data.append(words)

    #except:
        #print(f"Failed to fetch book with ID {book_id}")

# Concatenate all the tokenized words from all the books into a single list
all_words = [word for book_words in book_data for word in book_words]

# train the model on all the words from all the books 
model = Word2Vec(sentences=[all_words], vector_size=100, min_count=1, workers=4)

# Get the vector representation of a word
# similar_words = model.wv.most_similar('african', topn=10)
# print(similar_words)

vector = model.wv["black"]

# Convert the word vector to a numpy array
vector = np.asarray(vector)

# Create a pandas DataFrame with the word vector
df = pd.DataFrame(vector)

# Export the DataFrame to a CSV file
df.to_csv('black_vector.csv', index=False, header=False)

'''
# Get the vocabulary and the word vectors
vocab = model.vocab.keys()
vectors = [model[word] for word in vocab]

# Convert the word vectors to a numpy array
vectors = np.asarray(vectors)

# Create a pandas DataFrame with the vocabulary and the word vectors
df = pd.DataFrame(vectors, index=vocab)

# Export the DataFrame to a CSV file
df.to_csv('training_model_word_vectors.csv', index=True, header=False)
'''

'''
folder_path = 'text/'

# Define a function to read in the text from a file
def read_text(file_path):
    # use utf-8 encoding
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return f.read()


# Define a function to read in the text from all files in a folder
def read_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.gz'):
            file_path = os.path.join(folder_path, file_name)
            yield simple_preprocess(read_text(file_path))

# Create a generator for the text in the folder
text_generator = read_folder(folder_path)

# Train the Word2Vec model on the text
model = Word2Vec(text_generator, min_count=1, vector_size=100, workers=4)
similar_words = model.wv.most_similar('black', topn=10)
print(similar_words)

# Save the trained model to a file
# model.save('word2vec.model')

'''
