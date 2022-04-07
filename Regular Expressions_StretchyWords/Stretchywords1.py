
###
# Assignment stretchy  words 
# - function (charcountPerWord) count the number of each char in a word
# - function (findstretchyWords) compare the S word and query word list to find if it is stretchy 
# - In the MAIN the input S and queryWords can be changed(any list and string) for finding sterchy words of S that they are in the queryWords list  
# - in the end it prints the allTrueResult that is a tuple (stretchy word ,[each character , number of that character which is more than normal in that word])
# - and Also the number of strechy words in S (len of allTrueResult)
# - for test  (queryWords and s are changeable in the main )
#    queryWords={'hello','hi','why','yes'}    
#    s='heeelloooo hiiii  whyyyy  helo'  #inputs queryWords and s are changeable
# - output number of stretchy  words is:  3
# - dataframe stretchy  words :
# -[('heeelloooo', [('h', 0), ('e', 2), ('l', 0), ('o', 3)]), ('hiiii', [('h', 0), ('i', 3)]), ('whyyyy', [('w', 0), ('h', 0), ('y', 3)])]
###
import re
# count the number of each char of a word 
def charcountPerWord(word):
            chCount = []
            i = 0
            while i < len(word):
                char = word[i]
                start = i
                while i < len(word) and word[i] == char:
                    i += 1
                chCount.append((char, i-start))

            return chCount
# compare the S word and query word list to find if it is stretchy 
def findstretchyWords(Scount, wordCount):
            
            stretchyChars=[]
            if len(Scount) != len(wordCount):
                stretchyChars.append('Not same word')
                return False,stretchyChars
            for i in range(len(Scount)):
                if Scount[i][0] != wordCount[i][0]:
                    stretchyChars.append('Not same word')
                    return False,stretchyChars
                if Scount[i][1] < wordCount[i][1]:
                    stretchyChars.append('stretchy  char count is less than normal')
                    return False,stretchyChars
                if (Scount[i][1] > wordCount[i][1] and Scount[i][1] < 3):
                    stretchyChars.append('stretchy  char count is less than 3')
                    return False,stretchyChars
                stretchyChars.append((wordCount[i][0],Scount[i][1]-wordCount[i][1]))  # find the streached char,save the (char,streached len)
            return True,stretchyChars


if __name__ == '__main__':

 queryWords={'hello','hi','why','yes'}      #inputs queryWords and s are changeable
 s='heeelloooo hiiii  whyyyy  helo'
 
 resultWordQuery=[]
 resultString=[]
 # extract number of each char of the query list words and save in resultWordQuery tuple
 for word in queryWords:        
    resultWordQuery.append((word,charcountPerWord(word)))   
#  print(resultWordQuery)
 
  # extract number of each char of the S words and save in resultString tuple
 wordsOfS=re.findall(r"\w+",s)
 for wordS in wordsOfS:    
    resultString.append((wordS,charcountPerWord(wordS)))
#  print(resultString)

 # find strechy word and print output
 countresult=[]   
 allTrueResult=[]
 s=w=0  
 for s in range(len(resultString)):
    for w in range(len(resultWordQuery)):
        rsult,countresult=findstretchyWords(resultString[s][1],resultWordQuery[w][1])    
        if rsult==True:            
            allTrueResult.append((resultString[s][0],countresult))  # just the true stretchy words are saved in the allTrueResult tuple

 print("\n number of stretchy  words is: ",len(allTrueResult))
 print("\n dataframe stretchy  words :\n ",allTrueResult)


