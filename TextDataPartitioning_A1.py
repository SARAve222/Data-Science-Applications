
###
# TextDataPartitoning 
# -I 've used nltk for downloading books from gutenburg,and access them,It could also be
# done by download them one by one and  used file methods to open them and create sting of books 
# -I've used re.findall to  separate  all the words of each text,then united 100 words as a partition.
# -I've created 200 samples for each book and per book I have a label(A,B,C,..)
# -I've used pandas for serialize the output and make the .csv and .json files
# and used Both pd.series and pd.dataframe for label the output.I found that Dataframe is more clearer
###
import nltk 
import re
from nltk.corpus import gutenberg
import pandas as pd
from random import sample 

# nltk.download()     for downloading books of gutenburg 
files= gutenberg.fileids()
book_sapmle=[]
for book in files:
    book_str = gutenberg.raw(book)   
    book_words = re.findall(r"\w+", book_str)
    # print(book_words)
    Excactly100words=len(book_words)-len(book_words)%100    
    grouped_100words=[]
    for i in range(0, Excactly100words, 100):
        grouped_100words.append([' '.join(book_words[i: i + 100])])
    if(len(grouped_100words)>200):
     num_samples=200
    else:
     num_samples=len(grouped_100words)
    book_sapmle.append(sample(grouped_100words,num_samples))
    # print(book_sapmle)

books_samples_series=pd.Series(book_sapmle,index=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'])
books_samples_series.to_csv('gutenbergBooksSeries.csv', index=True)
# books_samples_series.to_csv('gutenbergBooksSeries.txt', sep=' ', index=True)
books_samples_series.to_json('gutenbergBookSeries.json',orient="index",indent=2)

books_samples_DF=pd.DataFrame(book_sapmle,index=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'])
books_samples_DF.to_json('gutenbergBooksDF.json',orient="index",indent=2)
books_samples_DF.to_csv('gutenbergBooksDF.csv', index=True)


