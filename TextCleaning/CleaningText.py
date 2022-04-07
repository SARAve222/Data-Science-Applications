import nltk
from nltk.corpus import stopwords
import re

def cleaning_stop_words(textWords):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w.lower() for w in textWords if not w.lower() in stop_words]   
    return filtered_sentence

###cleaning document
def cleaning_textToWord(rawString):
    text_words_no_sign=re.sub(r"[_ -]+", " ", rawString) 
    text_words_no_digit=re.sub(r'[0-9]+','',text_words_no_sign)
    text_words = re.findall(r"\w+", text_words_no_digit)
    number_of_words=len(text_words)
    cleanedtextWords=cleaning_stop_words(text_words)
    textfinal=' '.join(cleanedtextWords)
    return textfinal


