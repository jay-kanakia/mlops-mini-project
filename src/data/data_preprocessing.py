import pandas as pd
import numpy as np
import os
import re
import nltk
import string
import logging

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('data_preprocessing.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

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

def normalize_text(df:pd.DataFrame)->pd.DataFrame:
  try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug('converted to lower case')
        df['content'] = df['content'].apply(remove_url)
        logger.debug('url removed')
        df['content'] = df['content'].apply(remove_tag)
        logger.debug('tags removed')
        df['content'] = df['content'].apply(remove_punc)
        logger.debug('punctuations removed')
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('stopwords removed')
        df['content'] = df['content'].apply(stemming)
        logger.debug('stemming performed')
        df['content'] = df['content'].apply(is_alpa_num)
        logger.debug('is alpha numeric checked')
        logger.debug('Text normalization completed')
        return df
  except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def load_data(data_path:str)->tuple[pd.DataFrame,pd.DataFrame]:
  try:
    raw_data_path=os.path.join(data_path,'raw')
    train_data=pd.read_csv(os.path.join(raw_data_path,'train.csv'))
    test_data=pd.read_csv(os.path.join(raw_data_path,'test.csv'))
    logger.debug('data loaded successfully from path %s',data_path)
    return train_data,test_data
  except Exception as e:
     logger.error('Unexpected error occured during loading data %s',e)


def pre_process(train_data:pd.DataFrame,test_data:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
  try:
    train_data=normalize_text(train_data)
    test_data=normalize_text(test_data)
    logger.debug('data preprocessing completed')
    return train_data,test_data
  except Exception as e:
    logger.error('Unexpected error occured while preprocessing the data')
    raise
  
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
  try:
    raw_data_path=os.path.join(data_path,'interim')
    os.makedirs(raw_data_path,exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path,'train_processed.csv'))
    test_data.to_csv(os.path.join(raw_data_path,'test_processed.csv'))
    logger.debug('Data saved successfully')
  except Exception as e:
    logger.error('Unexpected error occured while saving the data')
    raise
  
def main():
  try:
    train_data,test_data=load_data(data_path='./data')
    logger.debug('Data loaded successfully')
    train_data,test_data=pre_process(train_data,test_data)
    logger.debug('data preprocessed successfully')
    save_data(train_data,test_data,data_path='./data')
    logger.debug('data saved successfully')
  except Exception as e:
    logger.error('Unexpected error occured during Data preprocessing')
    raise
    

if __name__=='__main__':
  main()