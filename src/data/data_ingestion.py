import numpy as np
import pandas as pd
import os
import logging
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('error.log')
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
    
def load_data(url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.debug('Data loaded successfully from %s',url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data %s',e)
        raise

def data_preprocess(df:pd.DataFrame,params:dict)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        test_size=params['data_ingestion']['test_size']
        df.drop(columns=['tweet_id'],inplace=True)
        df = df[df['sentiment'].isin(['happiness','sadness'])]
        train_data,test_data=train_test_split(df,test_size=test_size,random_state=42)
        le=LabelEncoder()
        train_data['sentiment']=le.fit_transform(train_data['sentiment'].values)
        test_data['sentiment']=le.transform(test_data['sentiment'].values)
        logger.debug('Data preprocessing completed')
        return train_data,test_data
    except KeyError as e:
        logger.error('Missing column in the dataframe %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during preprocessing %s',e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('data saved successfully to path %s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured during saving data %s',e)

def main():
    try:
        df1=load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        params=load_param(param_path='./params.yaml')

        train_data,test_data=data_preprocess(df1,params)

        save_data(train_data,test_data,data_path='./data')
        logger.debug('Data ingestion process completed')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process %s',e)
        raise

if __name__=='__main__':
    main()