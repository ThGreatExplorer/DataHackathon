import csv
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

filename = "word_vectors.csv"

# Open the CSV file
with open(filename, 'r') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    
    # Initialize an empty list to hold the numbers
    numbers = []
    
    # Loop through each row in the CSV file
    for row in reader:
        #print("row")
        # Loop through each item in the row
        for item in row:
            # If the item is a number, add it to the list
            if item[2].isnumeric():
                numbers.append(item)

splitnums = []
for row in numbers:
    splitnums.append(row.split(','))
#print(len(means))

arr = np.genfromtxt(filename,
                    float,
                    skip_header=1,
                    delimiter=',',
                    usecols=(range(2,len(splitnums[0]))))
means = np.mean(arr,axis=1)
print(len(arr))
#print(means)

#----------------------------------------------------------------------------------------------------------------------------
#Cosine Similarity
def cossim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    #cosine = cosine_similarity(A,B)
    return cosine
# ---------------------------------------------------------------------------------------------------------------------------
#Calculating WEAT Score
def calcWEAT(array):
    fourparts = np.array_split(array,4)
    T1 = fourparts[0]
    T2 = fourparts[1]
    A1 = fourparts[2]
    A2 = fourparts[3]
    #print(len(T1))
    #print(len(T2))
    #print(len(A1))
    #print(len(A2))
    #WEAT = (mean(cos_sim(T1,A1)) - mean(cos_sim(T1,A2)) - mean(cos_sim(T2,A1)) + mean(cos_sim(T2,A2))) / std(cos_sim(T1,A1,T1,A2,T2,A1,T2,A2))
    weat = ((np.mean(cossim(T1,A1))) - (np.mean(cossim(T1,A2))) - (np.mean(cossim(T2,A1))) + (np.mean(cossim(T2,A2)))) / (np.std(np.array([cossim(T1,A1),cossim(T1,A2),cossim(T2,A1),cossim(T2,A2)])))
    return weat

print(calcWEAT(means))