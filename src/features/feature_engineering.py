import pandas as pd
import numpy as np
import os
import logging
import yaml
import pickle

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

logger=logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('model_building.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(param_path:str)->dict:
    try:
        with open(param_path,'r') as file:
            params=yaml.safe_load(file)
            logger.debug('params loaded successfully')
            return params
    except FileNotFoundError:
        logger.error('File not found at give path %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured %s',e)
        raise

def load_data(data_path):
    try:
        raw_data_path=os.path.join(data_path,'interim')
        train_data=pd.read_csv(os.path.join(raw_data_path,'train_processed.csv'))
        train_data.fillna('', inplace=True)
        test_data=pd.read_csv(os.path.join(raw_data_path,'test_processed.csv'))
        test_data.fillna('',inplace=True)
        logger.debug('Data loaded successfully from url %s',data_path)
        return train_data,test_data
    except Exception as e:
        logger.error('Unexpected error occured during loading data from url %s',e)

def pre_process(train_data,test_data,params):
    try:
        X_train=train_data['content'].values
        y_train=train_data['sentiment'].values
        X_test=test_data['content'].values
        y_test=test_data['sentiment'].values

        max_features=params['feature_engineering']['max_features']
        cv=CountVectorizer(max_features=max_features)
        X_train_tfidf=cv.fit_transform(X_train)
        X_test_tfidf=cv.transform(X_test)

        os.makedirs('./models',exist_ok=True)
        with open ('./models/vectorizer.pkl','wb') as file:
            pickle.dump(cv,file)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test
        logger.debug('data preprocessing completed')
        return train_df,test_df
    except Exception as e:
        logger.error('Unexpected error occured during data preprocessing %s',e)
        raise

def save_data(train_df,test_df,data_path):
    try:
        raw_data_path=os.path.join(data_path,'processed')
        os.makedirs(raw_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path,'train_bow.csv'))
        test_df.to_csv(os.path.join(raw_data_path,'test_bow.csv'))
        logger.debug('Data saved successfully as %s',data_path)
    except Exception as e:
        logger.error('Unexpected error occured during saving the data %s',e)
        raise

def main():
    try:
        train_data,test_data=load_data(data_path='./data/')
        params=load_param(param_path='./params.yaml')
        train_df,test_df=pre_process(train_data,test_data,params)
        save_data(train_df,test_df,data_path='./data')
        logger.debug('Feature engineering step completed successfully')
    except Exception as e:
        logger.error('Some error occured during feature engineering step %s',e)

if __name__=='__main__':
    main()


