import pandas as pd
import numpy as np
import os
import pickle
import json
import logging

from sklearn.metrics import accuracy_score,f1_score,precision_score,roc_auc_score,classification_report,recall_score
from sklearn.linear_model import LogisticRegression

logger=logging.getLogger('Model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('model_eval.log')
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str)->LogisticRegression:
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
            logger.debug('Model loaded successfully')
        return model
    except Exception as e:
        logger.error('Unexpected error occured during loading model')
        raise

def load_data(data_path:str)->tuple[np.ndarray,np.ndarray]:
    try:
        test_data=pd.read_csv(data_path)
        X_test=test_data.iloc[:,:-1]
        y_test=test_data.iloc[:,-1]
        logger.debug('Test data loaded successfully')
        return X_test,y_test
    except Exception as e:
        logger.error('Some unexoected error has been occured %s',e)
        raise

def model_eval(model:LogisticRegression,X_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred=model.predict(X_test)
        y_pred_proba=model.predict_proba(X_test)[:,1]

        acc=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics_dict={
            'accuracy_score' : acc,
            'precision_score':precision,
            'recall_score':recall,
            'roc_auc_score':auc
        }
        logger.debug('Metrics calculated successfully')
        return metrics_dict
    except Exception as e:
        logger.error('some error occired during evaluation of metrics %s',e)
        raise

def save_metrics(metrics_dict:dict,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics_dict,file,indent=4)
            logger.debug('JSON file loaded successfully')
    except Exception as e:
        logger.error('Some error occured during loading of JSON file %s',e)

def main():
    try:
        model=load_model(file_path='./models/model.pkl')
        X_test,y_test=load_data(data_path='./data/processed/test_bow.csv')
        metrics_dict=model_eval(model,X_test,y_test)
        save_metrics(metrics_dict,file_path='./reports/metrics.json')
        logger.debug('Model evaluation step completed successfully')
    except Exception as e:
        logger.error('Some error occured during Model evaluation step %s',e)
        raise

if __name__=='__main__':
    main()