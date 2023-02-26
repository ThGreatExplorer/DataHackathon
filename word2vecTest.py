from gensim.models import Word2Vec
import csv

# TODO: Replace this with the dataset library
corpus = [["the", "quick", "brown", "fox"], ["jumped", "over", "the", "lazy", "dog"]]

# TODO: Replace this with better-suited paramenters
'''
From Chatgpt:
vector_size: This parameter determines the dimensionality of the word vectors that the model learns. A typical value for this parameter is 100-300. A higher value will result in more accurate word vectors, but will also require more memory and computational resources.

window: This parameter determines the size of the context window used during training. A typical value is 5-10 words. A larger window size will capture more semantic relationships between words, but may also lead to more noise in the training data.

min_count: This parameter determines the minimum frequency of a word to be included in the vocabulary. A typical value is 5-10 occurrences. Setting this value higher will result in a smaller vocabulary and faster training time, but may also result in the loss of rare words.

sg (skip-gram) vs cbow (continuous bag-of-words): These are the two algorithms that Word2Vec uses for training. Skip-gram is typically better suited for larger datasets, while CBOW is faster and works better for smaller datasets.

negative and sample: These parameters determine the way negative sampling is used during training. Negative sampling helps the model to distinguish between words that appear together frequently and words that do not. A typical value for negative is 5-20, and a typical value for sample is 1e-3 to 1e-5.
'''

model = Word2Vec(vector_size=100, min_count=1, workers=4)

# Build the vocabulary from the corpus
model.build_vocab(corpus)

# Train the model on the corpus
model.train(corpus, total_examples=model.corpus_count, epochs=10)

# Get the vector representation of a word
vector = model.wv["fox"]
print(vector)

model.wv.save_word2vec_format('word_vector.txt', binary=False)