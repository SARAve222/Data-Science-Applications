
from inspect import indentsize
import nltk 
import re
import pandas as pd

from random import sample

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

# nltk.download('punkt')
# nltk.download('stopwords')

###
### function: Title_Author_gutenburgbook
### all the gutenburg books downloaded from site have the Title and Author name in the same 
###  postion of the text.   
###
def extract_Title_Author_gutenburgbook(sentenceIncludeAuthor):
    bookTitle=""
    bookAuthor=""
    splitsent=sentenceIncludeAuthor.splitlines()
    if (('Title' and 'Author') in sentenceIncludeAuthor):
        test=sentenceIncludeAuthor.splitlines()       
        bookTitle=splitsent[0][7:]
        bookAuthor=splitsent[2][8:]
    return bookTitle,bookAuthor

###
### cleaning_stop_words: using stop_words fromnltk.corpus to clean the words.
###
def cleaning_stop_words(textWords):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w.lower() for w in textWords if not w.lower() in stop_words]   
    return filtered_sentence

###
### sampling_words: taking rawstring of the book, find the words using Regex
### cleaning the words by cleaning_stop_words()
###  geting numberof words and nuber of samples from input
###  usnig random.sample for choose the n sample
###
def sampling_words(rawStringOfbook,wordNum=100,SampleNum=200):
    book_words = re.findall(r"\w+", rawStringOfbook)
    cleanedBookWords=cleaning_stop_words(book_words)
    # print(cleanedBookWords)
    # print(book_words)
    Excactly100words=len(cleanedBookWords)-len(cleanedBookWords)% wordNum    
    grouped_100words=[]
    for i in range(0, Excactly100words, wordNum):
        grouped_100words.append(' '.join(cleanedBookWords[i: i + wordNum]))
    if(len(grouped_100words)>SampleNum):
     num_samples=SampleNum
    else:
     num_samples=len(grouped_100words)
    sampleOftheBook=sample(grouped_100words,num_samples)
    # print(sampleOftheBook)
    return sampleOftheBook


################

def Feature_Engineering_BOW(AllDataFrame):

    # # split the dataset into training and test datasets 
    train_x_BOW, test_x_BOW, train_y_BOW, test_y_BOW = model_selection.train_test_split(AllDataFrame["sampleText"],AllDataFrame["Authors"],train_size = 0.33, random_state=53)
    
    bow_vect = CountVectorizer(ngram_range=(1, 2), analyzer='word',max_features=5000 ,lowercase=False)
    bow_vect.fit(train_x_BOW)
    # transform data using count vectorizer object
    train_x_BOW_result = bow_vect.transform(train_x_BOW)
    # print for checking
    # print("\n\n")
    # print(train_x_BOW_result.toarray())
    # print("\n\n")
    # x_feature=bow_vect.get_feature_names_out()
    # x_feature_DF=pd.DataFrame(train_x_BOW_result.toarray(),columns=x_feature)
    # print("\n\n")
    # print(x_feature_DF)
    # print("\n\n")

    test_x_BOW_result=bow_vect.transform(test_x_BOW)

    encoder = preprocessing.LabelEncoder()
    train_y_BOW_result = encoder.fit_transform(train_y_BOW)
    test_y_BOW_result = encoder.fit_transform(test_y_BOW)   

    return train_x_BOW_result, test_x_BOW_result, train_y_BOW_result, test_y_BOW_result


def Feature_Engineering_TFIDF(AllDataFrame):

    # # split the dataset into training and test datasets 
    train_x_TFIDF, test_x_TFIDF, train_y_TFIDF, test_y_TFIDF = model_selection.train_test_split(AllDataFrame["sampleText"],AllDataFrame["Authors"],train_size = 0.33, random_state=53)
    
    tfidf_vect = TfidfVectorizer(analyzer='word',  ngram_range=(1,2), max_features=5000)
    tfidf_vect.fit(train_x_TFIDF)
    # transform data using TFIDF vectorizer object
    train_x_TFIDF_result =tfidf_vect.transform(train_x_TFIDF)
    # print for checking
    # print("\n\n")
    # print(train_x_TFIDF_result.toarray())
    # print("\n\n")
    # x_feature=tfidf_vect.get_feature_names_out()
    # x_feature_DF=pd.DataFrame(train_x_TFIDF_result.toarray(),columns=x_feature)
    # print("\n\n")
    # print(x_feature_DF)
    # print("\n\n")

    test_x_TFIDF_result =tfidf_vect.transform(test_x_TFIDF)

    encoder = preprocessing.LabelEncoder()
    train_y_TFIDF_result = encoder.fit_transform(train_y_TFIDF)
    test_y_TFIDF_result = encoder.fit_transform(test_y_TFIDF)
    return train_x_TFIDF_result, test_x_TFIDF_result, train_y_TFIDF_result, test_y_TFIDF_result


# #### Modeling
# sclassifier_Method :svm.SVC() ,DecisionTreeClassifier(),KNeighborsClassifier,ensemble.RandomForestClassifier()
def train_modeling(classifier_Method, vector_train_x, vector_train_y, vector_test_x,vector_test_y):
    # fit the training dataset on the classifier
    ouput=classifier_Method.fit(vector_train_x, vector_train_y)  
    # predict the labels on validation dataset
    predictions = classifier_Method.predict(vector_test_x) 
    accuracymetic= metrics.accuracy_score(predictions, vector_test_y)
    report=metrics.classification_report(predictions, vector_test_y)
    matrix_conf= metrics.confusion_matrix (predictions,vector_test_y)
    return accuracymetic ,matrix_conf,report,predictions

def Matrix_conf_Display(matrixConfResult):
    mpl.interactive(1)
   #Confusion Matrix
    ax = sns.heatmap(matrixConfResult, annot=True, cmap='Blues')
    # ax = sns.heatmap(matrix_KNeighbors_TFIDF/np.sum(matrix_KNeighbors_TFIDF), annot=True, 
    #         fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Display the visualization of the Confusion Matrix.
    plt.show()
############ MAIN ################
if __name__ == '__main__':

    files = ".*\.txt"
    corpus_root='C:\Python\Assignment2\Books'
    corpus0 = PlaintextCorpusReader(corpus_root, files)
    files=corpus0.fileids()
    book_sapmle=[]
    bookInfo=[]
    titles=[]
    authors=[]
    num_samples=200
    for book in files:
        book_str = corpus0.raw(book)
        tokenSent =sent_tokenize(book_str)  
        bookInfo.append(extract_Title_Author_gutenburgbook(tokenSent[3]) )
        book_sapmle.append(sampling_words(book_str,wordNum=100,SampleNum=200))
        bookTitle,bookAuthor=(extract_Title_Author_gutenburgbook(tokenSent[3]) )
        titles.append(bookTitle)
        authors.append(bookAuthor)
   
    books_samples_bookInfo_DF=pd.DataFrame(book_sapmle,bookInfo)
    books_samples_bookInfo_DF.to_json('gutenbergBooksDFInfo.json',orient="index",indent=2)

    books_samples_authors_DF=pd.DataFrame(book_sapmle,authors)
   # books_samples_authors_DF.to_json('gutenbergBooksDF.json',orient="index",indent=2)

#########prepare Data for tran and test

    tempSample=[]
    tempauthors=[]
    for k in range(0,book_sapmle.__len__()):
        for sampleindex in book_sapmle[k]: 
            tempSample.append(sampleindex)
            tempauthors.append(authors[k])
   

    allSampleforTesTrian=pd.DataFrame()
    allSampleforTesTrian["sampleText"]=tempSample
    allSampleforTesTrian["Authors"]=tempauthors

    shuffle(allSampleforTesTrian)
    allSampleSuffle=pd.DataFrame(allSampleforTesTrian)
    # print(allSampleSuffle)
    # allSampleSuffle.to_csv('allSampleforTesTrianDF.csv',index=True)

#################################
    train_x_BOW, test_x_BOW, train_y_BOW, test_y_BOW=Feature_Engineering_BOW(allSampleSuffle)
    # print(train_x_BOW.toarray())
    train_x_TFIDF, test_x_TFIDF, train_y_TFIDF, test_y_TFIDF= Feature_Engineering_TFIDF(allSampleSuffle)
    # print(train_x_TFIDF.toarray())


    ####use modeling 
    accuracy_svm_BOW,matrix_svm_BOW,report_svm_BOW,predictions_svm_BOW=train_modeling(svm.SVC(), train_x_BOW, train_y_BOW, test_x_BOW,test_y_BOW)
    accuracy_svm_TFIDF,matrix_svm_TFID,report_svm_TFID,predictions_svm_TFIDF=train_modeling(svm.SVC(), train_x_TFIDF, train_y_TFIDF, test_x_TFIDF,test_y_TFIDF)

    accuracy_DecisionTree_BOW,matric_DecisionTree_BOW,report_DecisionTree_BOW,predictions_DecisionTree_BOW=train_modeling(DecisionTreeClassifier(), train_x_BOW, train_y_BOW, test_x_BOW,test_y_BOW)
    accuracy_DecisionTree_TFIDF,matric_DecisionTree_TFIDF,report_DecisionTree_TFIDF,predictions_DecisionTree_BOW=train_modeling(DecisionTreeClassifier(), train_x_TFIDF, train_y_TFIDF, test_x_TFIDF,test_y_TFIDF)

    accuracy_KNeighbors_BOW,matrix_KNeighbors_BOW,report_KNeighbors_BOW,predictions_KNeighbors_BOW=train_modeling(KNeighborsClassifier(), train_x_BOW, train_y_BOW, test_x_BOW,test_y_BOW)
    accuracy_KNeighbors_TFIDF,matrix_KNeighbors_TFIDF,report_KNeighbors_TFIDF,predictions_KNeighbors_TFIDF=train_modeling(KNeighborsClassifier(), train_x_TFIDF, train_y_TFIDF, test_x_TFIDF,test_y_TFIDF)
     
    accuracy_RandomForest_BOW,matrix_RandomForest_BOW,report_RandomForest_BOW,predictions_RandomForest_BOW=train_modeling(ensemble.RandomForestClassifier(), train_x_BOW, train_y_BOW, test_x_BOW,test_y_BOW)
    accuracy_RandomForest_TFIDF,matrix_RandomForest_TFIDF,report_RandomForest_TFIDF,predictions_RandomForest_BOW=train_modeling(ensemble.RandomForestClassifier(), train_x_TFIDF, train_y_TFIDF, test_x_TFIDF,test_y_TFIDF)


    acuracyModelingDF=pd.DataFrame()
    acuracyModelingDF["Modeling"]=["SVM_BOW","SVM_TFIDF","DecisionTree_BOW","DecisionTree_TFIDF","KNeighbors_BOW","KNeighbors_TFIDF","RandomForest_BOW","RandomForest_TFIDF"]
    acuracyModelingDF["Accuracy"]=[accuracy_svm_BOW,accuracy_svm_TFIDF,accuracy_DecisionTree_BOW,accuracy_DecisionTree_TFIDF,accuracy_KNeighbors_BOW,accuracy_KNeighbors_TFIDF,accuracy_RandomForest_BOW,accuracy_RandomForest_TFIDF]
   
    print("\n\n")
    print(acuracyModelingDF)
    print("\n\n")
    acuracyModelingDF.to_csv('acuracyModelingDF.csv', index=True)


    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Modeling", y="Accuracy", data=acuracyModelingDF)
    plt.xticks(rotation=90)


    #########Select the champion and display matrix conf /report /Error Analysis  :KNeighbors_TFIDF
    Matrix_conf_Display(matrix_KNeighbors_TFIDF)    
    print(report_KNeighbors_TFIDF)

    ####Error Analysis
    errorAnalysis=pd.DataFrame(columns=('Actual Author', 'predictedAuthor' ,'Cases'))
    counter=0
    for predicted in range(0,authors.__len__()):
	    for actual in range(0,authors.__len__()):
             if(matrix_KNeighbors_TFIDF[actual,predicted]>=1 and predicted!=actual):
                    print(f"In {matrix_KNeighbors_TFIDF[actual,predicted]} cases  the actual author is '{authors[test_y_TFIDF[actual]]}' but it predicted as '{authors[predictions_KNeighbors_TFIDF[predicted]]}' !")     
                    errorAnalysis.loc[counter]=[authors[test_y_TFIDF[actual]] ,authors[predictions_KNeighbors_TFIDF[predicted]],matrix_KNeighbors_TFIDF[actual,predicted]]
                    counter=counter+1

    errorAnalysis.to_csv('errorAnalysis.csv', index=True) 
    print(errorAnalysis)              
######################################
   
    



