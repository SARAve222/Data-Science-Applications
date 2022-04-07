1-  "Text_Classification1.py" : The fallwoing libraries are needed

1-1: re
1-2: nltk
1-3: pandas
1-4: random
1-5: sklearn
1-6: matplotlib
1-7: seaborn
   
   
2- The books are downloaded in the books folder and in the code the path is "corpus_root='C:\Python\Assignment2\Books'"
   please change root of the book to your root. The book folder is attached.
   
3-program has this output files
 3-1: gutenbergBooksDFInfo.json
     all 200 samples of each book,labeling by authors and titles
	 
 3-2: acuracyModelingDF.csv
	  the df of the acuracy of each model 
 
 3-3:errorAnalysis.csv
     The df of  champion model in cases that autual author and the predicted are not same
	 
 3-4: plot a bar chart of modeling "ModelingcompareFinal.png" is attached    
 
 3-5: plot a Confusion matrix "ConfusionMatrixFinal.png" is attaced
     th Confusion matrix of the champion model
 
 
 
 
4-most of the print commands are commented.these are whole print outputs that are used for report.



             Modeling  Accuracy
0             SVM_BOW  0.837248
1           SVM_TFIDF  0.837248
2    DecisionTree_BOW  0.704698
3  DecisionTree_TFIDF  0.689597
4      KNeighbors_BOW  0.595638
5    KNeighbors_TFIDF  0.914430
6    RandomForest_BOW  0.859060
7  RandomForest_TFIDF  0.877517



              precision    recall  f1-score   support

           0       1.00      0.86      0.92       162
           1       0.94      0.87      0.90       133
           2       0.87      0.97      0.91       122
           3       0.90      0.96      0.93       119
           4       0.83      0.97      0.89        60

    accuracy                           0.91       596
   macro avg       0.91      0.92      0.91       596
weighted avg       0.92      0.91      0.91       596

In 6 cases  the actual author is 'Walter Besant' but it predicted as 'Miguel de Cervantes Saavedra' !
In 2 cases  the actual author is 'Leo Tolstoy' but it predicted as 'Miguel de Cervantes Saavedra' !
In 8 cases  the actual author is 'Walter Besant' but it predicted as 'Walter Besant' !
In 9 cases  the actual author is 'Miguel de Cervantes Saavedra' but it predicted as 'Walter Besant' !
In 1 cases  the actual author is 'Leo Tolstoy' but it predicted as 'Walter Besant' !
In 3 cases  the actual author is 'Walter Besant' but it predicted as 'Leo Tolstoy' !
In 6 cases  the actual author is 'Miguel de Cervantes Saavedra' but it predicted as 'Leo Tolstoy' !
In 2 cases  the actual author is 'Walter Besant' but it predicted as 'Leo Tolstoy' !
In 2 cases  the actual author is 'Fyodor Dostoevsky' but it predicted as 'Leo Tolstoy' !
In 6 cases  the actual author is 'Walter Besant' but it predicted as 'Fyodor Dostoevsky' !
In 2 cases  the actual author is 'Miguel de Cervantes Saavedra' but it predicted as 'Fyodor Dostoevsky' !
In 2 cases  the actual author is 'Walter Besant' but it predicted as 'Fyodor Dostoevsky' !
In 2 cases  the actual author is 'Leo Tolstoy' but it predicted as 'Fyodor Dostoevsky' !


                   Actual Author               predictedAuthor Cases
0                  Walter Besant  Miguel de Cervantes Saavedra     6
1                    Leo Tolstoy  Miguel de Cervantes Saavedra     2
2                  Walter Besant                 Walter Besant     8
3   Miguel de Cervantes Saavedra                 Walter Besant     9
4                    Leo Tolstoy                 Walter Besant     1
5                  Walter Besant                   Leo Tolstoy     3
6   Miguel de Cervantes Saavedra                   Leo Tolstoy     6
7                  Walter Besant                   Leo Tolstoy     2
8              Fyodor Dostoevsky                   Leo Tolstoy     2
9                  Walter Besant             Fyodor Dostoevsky     6
10  Miguel de Cervantes Saavedra             Fyodor Dostoevsky     2
11                 Walter Besant             Fyodor Dostoevsky     2
12                   Leo Tolstoy             Fyodor Dostoevsky     2


