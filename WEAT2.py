import csv
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# List of filenames
filenames = [
    "Data/biased_trainedmodel_word_vectors.csv", 
    "Data/unbiased_trainedmodel_word_vectors.csv", 
    "Data/all_trainedmodel_word_vectors.csv",
    "Data\GoogleNews_word_vectors.csv",
    ]

#----------------------------------------------------------------------------------------------------------------------------
#Extract Positive Words
positivefile = "positive-words.txt"

posarr = np.genfromtxt(positivefile,
                    str,
                    skip_header=35,
                    delimiter="\n",
                    usecols=(0))

#----------------------------------------------------------------------------------------------------------------------------
#Extract Negative Words
negativefile = "negative-words.txt"

negarr = np.genfromtxt(negativefile,
                    str,
                    skip_header=35,
                    delimiter="\n",
                    usecols=(0))

#----------------------------------------------------------------------------------------------------------------------------
#Templates for Contentious Words
#These words are input into the program to find whether they are associated with more positive or negative words

#female associated words
femalewords = ["women","daughter","mother","grandmother","nurse","girl","dress","skirt"]



#---------------------------------------------------------------------------------------------------------------------------
# Loop through each filename
for filename in filenames:
    print("File: ", filename)
    
    # Open the CSV file
    # need errors ignore for the GoogleNews csv file which has invalid characters from news articles
    with open(filename, 'r', encoding="utf-8", errors='ignore') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Initialize an empty list to hold the numbers
        numbers = []

        # Loop through each row in the CSV file
        for row in reader:
            # Loop through each item in the row
            for item in row:
                # If the item is a number, add it to the list
                if (len(item) > 5 and item[3].isnumeric()):
                    numbers.append(item)
           
    splitnums = []
    for row in numbers:
        splitnums.append(row.split(','))

    # ignore the warninngs raised by the improperly formatted vectors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # invalid raise set to false which will emit warnings and skip the incorrect lines
        arr = np.genfromtxt(filename,
                                dtype=str,
                                skip_header=1,
                                delimiter=',',
                                invalid_raise=False,
                                encoding="utf-8"
                                )

    print("TOTAL WORDS: ")
    print(len(arr))


    #Positive Word Vectors
    posvecarr = []
    for row in arr:
        if row[0] in posarr:
            posvecarr.append(row[2:100])


    posvecsfloats = (np.array(posvecarr)).astype(float)

    posmeans = np.mean(posvecsfloats,axis=1)
    print("WORDS SAMPLED: ")
    print(len(posmeans)*2)

    #Negative Word Vectors
    negvecarr = []
    countn = 0
    for row in arr:
        if row[0] in negarr and countn<len(posmeans):
            countn = countn+1
            negvecarr.append(row[2:100])


    negvecsfloats = (np.array(negvecarr)).astype(float)

    negmeans = np.mean(negvecsfloats,axis=1)

    #Target Word Mean
    specificarr = []
    count=0
    for row in arr:
        if count<2*len(posmeans):
            count=count+1
            specificarr.append(row[2:100])

    #Target Word Vectors
    specificvecsfloats = (np.array(specificarr)).astype(float)
    means = np.mean(specificvecsfloats,axis=1)
    if (len(means)%2)!=0:
        means = means[0:len(means)-1]
        
    #----------------------------------------------------------------------------------------------------------------------------
    #Cosine Similarity
    def cossim(A,B):
        cosine = np.dot(A,B)/(norm(A)*norm(B))
        #cosine = np.sum(A*B, axis=1)/(norm(A, axis=0)*norm(B, axis=0))
        #cosine = cosine_similarity(A,B)
        return cosine
    # ---------------------------------------------------------------------------------------------------------------------------
    #Calculating WEAT Score
    def calcWEAT():
        halfmeans = np.split(means,2)
        T1 = halfmeans[0]
        T2 = halfmeans[1]
        A1 = posmeans
        A2 = negmeans
        #print(len(T1))
        #print(len(T2))
        #print(len(A1))
        #print(len(A2))
        #WEAT = (mean(cos_sim(T1,A1)) - mean(cos_sim(T1,A2)) - mean(cos_sim(T2,A1)) + mean(cos_sim(T2,A2))) / std(cos_sim(T1,A1,T1,A2,T2,A1,T2,A2))
        weat = ((np.mean(cossim(T1,A1))) - (np.mean(cossim(T1,A2))) - (np.mean(cossim(T2,A1))) + (np.mean(cossim(T2,A2)))) / (np.std(np.array([cossim(T1,A1),cossim(T1,A2),cossim(T2,A1),cossim(T2,A2)])))
        return weat/2
    #print(len(means))
    #print(posmeans)
    #print(negmeans)
    print("WEAT SCORE: ")
    print(calcWEAT())
