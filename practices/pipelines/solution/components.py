from kfp.v2.components.types.artifact_types import ClassificationMetrics
from kfp.v2.dsl.experimental import component, Output, Dataset, Model, Input
from typing import NamedTuple



@component(
    packages_to_install=['scikit-learn'],
    base_image='python:3.11'
)
def load_and_split_data(train_dataset: Output[Dataset], test_dataset: Output[Dataset]) -> NamedTuple('outputs', train_size=int, test_size=int):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import joblib

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    joblib.dump((X_train, y_train), train_dataset.path)
    joblib.dump((X_test, y_test), test_dataset.path)


    return NamedTuple('outputs', train_size=int, test_size=int)(len(X_train), len(X_test))
    

@component(
    packages_to_install=['scikit-learn'],
    base_image='python:3.11'
)
def train_model(train_dataset: Input[Dataset], model_artifact: Output[Model]):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    X_train, y_train = joblib.load(train_dataset.path)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True))
    ])
    
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_artifact.path)

@component(
    packages_to_install=[ 'scikit-learn'],
    base_image='python:3.11'
)
def evaluate_model(test_dataset: Input[Dataset], model_artifact: Input[Model], metrics: Output[ClassificationMetrics]):
    from sklearn.metrics import confusion_matrix
    import joblib

    X_test, y_test = joblib.load(test_dataset.path)
    model = joblib.load(model_artifact.path)

    y_pred = model.predict(X_test)
    
    confusion_mtx = confusion_matrix(y_test, y_pred)
    
    metrics.log_confusion_matrix(
        [str(i) for i in range(len(confusion_mtx))],
        confusion_mtx.tolist()
    )
       
    
@component(
    packages_to_install=['tensorflow==2.15.0', 'google-cloud-storage'],
    base_image='python:3.11'
)
def sklearn_minio_to_gcs(dataset: Input[Dataset], model_artifact: Output[Model], gcs_bucket: str) -> NamedTuple('outputs', dataset_gcs_uri=str,  model_gcs_uri=str):
    from google.cloud import storage
    import os

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(gcs_bucket)

    # Upload dataset
    destination_path = f"{os.path.basename(dataset.path)}"
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(dataset.path)

    dataset_gcs_uri = f"gs://{gcs_bucket}/{os.path.basename(dataset.path)}"
    model_gcs_uri = f"gs://{gcs_bucket}/{os.path.basename(model_artifact.path)}"
    
    print("Dataset URI: ", dataset_gcs_uri)
    print("Model URI: ", model_gcs_uri)
    
    return NamedTuple('outputs', dataset_gcs_uri=str, model_gcs_uri=str)(dataset_gcs_uri, model_gcs_uri)


@component(
    packages_to_install=['tensorflow==2.15.0'],
    base_image='python:3.11'
)
def tf_model_gcs_to_minio(model_gcs_uri: str, model_artifact: Output[Model]):
    import tensorflow as tf
  
    model = tf.keras.models.load_model(model_gcs_uri)

    model.save(model_artifact.path)
