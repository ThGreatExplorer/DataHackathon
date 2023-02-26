import pandas as pd
import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

# model to classify the books into bias or unbiased

book_df = pd.read_csv('training_data.csv')

nltk.download('punkt')
book_df['sentences'] = book_df['text'].apply(nltk.sent_tokenize)
book_df['words'] = book_df['sentences'].apply(lambda x: [nltk.word_tokenize(sent) for sent in x])

model = Word2Vec(book_df['words'], vector_size=100, window=5, min_count=5, workers=4, sg=1)

book_df['embeddings'] = book_df['words'].apply(lambda x: sum([model.wv[word] for word in x]))

X_train, X_val, y_train, y_val = train_test_split(book_df['embeddings'], book_df['label'], test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train.tolist(), y_train.tolist(), validation_data=(X_val.tolist(), y_val.tolist()), epochs=10, batch_size=32)


