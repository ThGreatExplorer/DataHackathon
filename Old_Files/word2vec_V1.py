import gutenbergpy.textget
from gensim.models import Word2Vec

# Get the raw text of the book from Gutenberg
raw_book = gutenbergpy.textget.get_text_by_id(2701)

# Remove the headers from the book
clean_book = gutenbergpy.textget.strip_headers(raw_book)

# Process the text to remove unwanted characters
processed_book = clean_book.decode('utf-8').replace("\n", " ").replace("\r", " ").replace(".", " ").replace(",", " ")

# Split the text into a list of words
word_list = processed_book.split()

# Train the Word2Vec model on the book
model = Word2Vec(sentences=[word_list], vector_size=100, min_count=1, workers=4)

# Get the vector representation of a word
similar_words = model.wv.most_similar('Dick', topn=10)
print(similar_words)
