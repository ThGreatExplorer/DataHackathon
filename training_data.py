import re
import gutenbergpy.textget
import pandas as pd

# Define the paths to the biased and unbiased book ID files
biased_file = 'Training_biased'
unbiased_file = 'Training_unbiased'

# Read in the book IDs from the biased and unbiased files
with open(biased_file, 'r') as f:
    biased_ids = re.findall('\d+', f.read())

with open(unbiased_file, 'r') as f:
    unbiased_ids = re.findall('\d+', f.read())

# Fetch the text for each book and label it as biased or unbiased
book_data = []

for book_id in biased_ids + unbiased_ids:
    try:
        raw_book = gutenbergpy.textget.get_text_by_id(int(book_id))
        clean_book = gutenbergpy.textget.strip_headers(raw_book)
        processed_book = clean_book.decode('utf-8').replace("\n", " ").replace("\r", " ").replace(".", " ").replace(",", " ")
        label = 1 if book_id in biased_ids else 0
        book_data.append({'text': processed_book, 'label': label})
    except:
        print(f"Failed to fetch book with ID {book_id}")

# Create a pandas dataframe with the book data
book_df = pd.DataFrame(book_data)

# Save the book dataframe to a CSV file
book_df.to_csv('training_data.csv', index=False)
