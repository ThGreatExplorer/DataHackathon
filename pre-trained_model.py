from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import gensim.downloader as api
import gensim
import csv

# download pre-trained google news word2vec algo
model = api.load('word2vec-google-news-300')
#takes a long time

similar_words = model.most_similar('african', topn=10)
print(similar_words)

# Export the vectors to a CSV file
with open("GoogleNews_word_vectors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Vector"])
    for word in model.index_to_key:
        vector = model[word]
        writer.writerow([word, ",".join([str(x) for x in vector])])