import pandas as pd
import numpy as np
import os
import xgboost
import pickle
import logging
import yaml

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

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

def load_params(param_path:str)->dict:
    try:
        with open(param_path,'r') as file:
            params=yaml.safe_load(file)
            logger.debug('params loaded succesfuuly from the path %s',param_path)
            return params
    except FileNotFoundError:
        logger.error('File not found at the given path %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured %s',e)
        raise

def load_data(data_path:str)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        raw_data_path=os.path.join(data_path,'processed')
        train_data=pd.read_csv(os.path.join(raw_data_path,'train_bow.csv'))
        X_train=train_data.iloc[:,:-1]
        y_train=train_data.iloc[:,-1]

        # no need for test data here
        #test_data=pd.read_csv(os.path.join(raw_data_path,'test_tfidf.csv'))
        logger.debug('Data loaded successfully from url %s',data_path)
        return X_train,y_train
    except Exception as e:
        logger.error('Unexpected error occured during data loading %s',e)
        raise

def model_eval(X_train:np.ndarray,y_train:np.ndarray,params:dict)->LogisticRegression:
    try:
        max_iter=params['model_building']['max_iter']
        l1_ratio=params['model_building']['l1_ratio']

        # xgb_model=XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
        # xgb_model.fit(X_train,y_train)

        lr=LogisticRegression(max_iter=max_iter,solver='saga',l1_ratio=l1_ratio,n_jobs=-1)
        lr.fit(X_train,y_train)

        logger.debug('model builded successfully')
        return lr
    except Exception as e:
        logger.error('Unexpected error occured during model building %s',e)
        raise

def save_model(model:LogisticRegression,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
            logger.debug('model loaded successfully to the path %s',file_path)
    except Exception as e:
        logger.error('Unexpected error occured during saving the model %s',e)
        raise

def main():
    try:
        X_train,y_train=load_data(data_path='./data')
        params=load_params(param_path='./params.yaml')
        model=model_eval(X_train,y_train,params)
        save_model(model,file_path='./models/model.pkl')
        logger.debug('Model building step completed successfully')
    except Exception as e:
        logger.error('some error occured during model building')
        raise

if __name__=='__main__':
    main()