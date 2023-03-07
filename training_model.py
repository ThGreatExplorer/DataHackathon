import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gutenbergpy.textget
import csv

# downloads stopwords
# run this the first time you run this program and then comment out again
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# We want to program the word2vec to extract cosine similarities between words
# biased and unbiased training data

biased_file = 'Training_biased'
unbiased_file = 'Training_unbiased'

# Read in the book IDs from the biased and unbiased files
with open(biased_file, 'r') as f:
    biased_ids = re.findall('\d+', f.read())

with open(unbiased_file, 'r') as f:
    unbiased_ids = re.findall('\d+', f.read())

#---------------------------------------------------------------------------------------------------------------
# biased books

biased_book_data = []

# modify to compare against biased_ids
for book_id in biased_ids:
    try:
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
        biased_book_data.append(words)

    except:
        print(f"Failed to fetch book with ID {book_id}")

# Concatenate all the tokenized words from all the biased books into a single list
biased_all_words = [word for book_words in biased_book_data for word in book_words]

# train the model on all the words from all the biased books 
biased_model = Word2Vec(sentences=[biased_all_words], vector_size=100, min_count=1, workers=4)

# Get the vector representation of a word
# similar_words = model.wv.most_similar('african', topn=10)
# print(similar_words)

biased_vector = biased_model.wv["black"]
print(biased_vector)

#---------------------------------------------------------------------------------------------------------------

# unbiased books

unbiased_book_data = []

# modify to compare against unbiased_ids
for book_id in unbiased_ids:
    try:
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
        unbiased_book_data.append(words)

    except:
        print(f"Failed to fetch book with ID {book_id}")

# Concatenate all the tokenized words from all the unbiased books into a single list
unbiased_all_words = [word for book_words in unbiased_book_data for word in book_words]

# train the model on all the words from all the unbiased books 
unbiased_model = Word2Vec(sentences=[unbiased_all_words], vector_size=100, min_count=1, workers=4)

# Get the vector representation of a word
# similar_words = model.wv.most_similar('african', topn=10)
# print(similar_words)

unbiased_vector = unbiased_model.wv["black"]
print(unbiased_vector)

#---------------------------------------------------------------------------------------------------------------
# unbiased books

all_book_data = []

# modify to compare against unbiased_ids
for book_id in unbiased_ids + biased_ids:
    try:
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
        all_book_data.append(words)

    except:
        print(f"Failed to fetch book with ID {book_id}")

# Concatenate all the tokenized words from all the biased books into a single list
all_all_words = [word for book_words in all_book_data for word in book_words]

# train the model on all the words from all the biased books 
all_model = Word2Vec(sentences=[all_all_words], vector_size=100, min_count=1, workers=4)

# Get the vector representation of a word
# similar_words = model.wv.most_similar('african', topn=10)
# print(similar_words)

all_vector = all_model.wv["black"]
print(all_vector)

#---------------------------------------------------------------------------------------------------------------
# Exporting word embeddings as csv

# Export the biased model's vectors to a CSV file
with open("Data/biased_trainedmodel_word_vectors.csv", "w", encoding='utf-8', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Vector"])
    for word in biased_model.wv.index_to_key:
        vector = biased_model.wv[word]
        writer.writerow([word, ",".join([str(x) for x in vector])])

# Export the unbiased model's vectors to a CSV file
with open("Data/unbiased_trainedmodel_word_vectors.csv", "w", encoding='utf-8', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Vector"])
    for word in unbiased_model.wv.index_to_key:
        vector = unbiased_model.wv[word]
        writer.writerow([word, ",".join([str(x) for x in vector])])

# Export the biased and unbiased training model's vectors to a CSV file
with open("Data/all_trainedmodel_word_vectors.csv", "w", encoding='utf-8', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Vector"])
    for word in all_model.wv.index_to_key:
        vector = all_model.wv[word]
        writer.writerow([word, ",".join([str(x) for x in vector])])
