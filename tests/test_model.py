# model test v1 - loading the model

import unittest
import mlflow
import os

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri('http://ec2-52-66-126-66.ap-south-1.compute.amazonaws.com:80')
        
        # Load the model from MLflow model registry
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

if __name__ == "__main__":
    unittest.main()