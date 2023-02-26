# DataHackathon
Working Prototype for Northeastern's Data Hackathon 2023

## Inspiration

**Inspired by the human biases in translation**, specifically how translator's bias could affect the meaning and interpretation of books and the original author's intention. We decided to create our project on the idea of detecting bias and in human and machine translations

## How we built it

We first curated our own dataset of biased and unbiased books based on factors such as sexism, racism, discrimination, and other forms of bias.

## Challenges we ran into

One challenge that influenced the progress of our project was the parsing of data and data format such that vector sizes and input shapes could be standardized so that WEAT analysis could be performed.

## Accomplishments that we're proud of

- curated own dataset of 40+ books, labeled 0 for unbias and 1 for bias
- created own Word2Vec model based on bias and unbiased samples and mapped word embeddings
- implemented a hard-coded WEAT analysis algorithm with a standard word sample of target and attribute words that outputs a metric for bias
- compared against benchmark of the  


