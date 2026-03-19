import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def lower_case(text:str)->str:
  text=text.lower()
  return text

# remove url

def remove_url(text:str)->str:
  pattern=re.compile(r'https?://\S+|www\.\S+')
  return pattern.sub(r'',text)

#remove html tag
def remove_tag(text:str)->str:
  pattern=re.compile(r'<.*?>')
  return pattern.sub(r'',text)

# remove punctuation
exclude=string.punctuation

def remove_punc(text:str)->str:
  text=[i for i in text if i not in exclude]
  return ''.join(text)

# remove stopwords
stop_words=set(stopwords.words('english'))

def remove_stop_words(text:str)->str:
  text=[i for i in text.split() if i not in stop_words]
  return ' '.join(text)

# lemmatization
ps=PorterStemmer()

def stemming(text:str)->str:
  text=[ps.stem(i) for i in text.split()]
  return ' '.join(text)

# isAlphaNum
def is_alpa_num(text:str)->str:
  text=[i for i in text.split() if i.isalnum()]
  return ' '.join(text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_url(text)
    text = remove_tag(text)
    text = remove_punc(text)
    text = remove_stop_words(text)
    text = stemming(text)
    text = is_alpa_num(text)

    return text