import os
import mlflow

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def promote_model():

    dagshub_token=os.getenv('DAGSHUB_PAT')
    if not dagshub_token:
        raise EnvironmentError('DAGSHUB_PAT environment variable is not set')

    os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_token

    mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/mlops-mini-project.mlflow')

    client=mlflow.tracking.MlflowClient()

    model_name='final_model'

    latest_version_staging=client.get_latest_versions(model_name,stages=['Staging'])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()