import gutenbergpy.textget

# TODO: find someway to get the gutenberg id numbers for a large set of books that also fulfill our requirements of translated/orginal copies
def usage_example():
    # This gets a book by its gutenberg id number
    raw_book = gutenbergpy.textget.get_text_by_id(2701) # with headers
    clean_book = gutenbergpy.textget.strip_headers(raw_book) # without headers
    return clean_book, raw_book

cleaned_book, raw_book = usage_example()

# Cleaned Book
# only outputs chars from 3000 to 3050
print(f'Example phrase from the cleaned book: {" ".join(str(cleaned_book[3000:3050]).split(" "))}')

# Raw Book
print(f'Example phrase from the raw book: {" ".join(str(raw_book).split(" "))}')
