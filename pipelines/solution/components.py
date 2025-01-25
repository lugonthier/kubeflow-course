from kfp.v2.components.types.artifact_types import ClassificationMetrics
from kfp.v2.dsl.experimental import component, Output, Dataset, Model, Input
from typing import NamedTuple



@component(
    packages_to_install=['tensorflow==2.15.0', 'tensorflow-datasets'],
    base_image='python:3.11'
)
def load_and_split_data(train_dataset: Output[Dataset], test_dataset: Output[Dataset]) -> NamedTuple('outputs', train_size=int, test_size=int):
    import tensorflow_datasets as tfds

    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
    
    ds_train.save(train_dataset.path)
    ds_test.save(test_dataset.path)

    return NamedTuple('outputs', train_size=int, test_size=int)(ds_info.splits['train'].num_examples, ds_info.splits['test'].num_examples)

@component(
    packages_to_install=['tensorflow==2.15.0'],
    base_image='python:3.11'
)
def preprocess_data(dataset: Input[Dataset], preprocessed_dataset: Output[Dataset], size: int):
    import tensorflow as tf
    
    ds = tf.data.Dataset.load(dataset.path)
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    ds = ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(size)
    ds = ds.batch(64)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds.save(preprocessed_dataset.path)
    

    

@component(
    packages_to_install=['tensorflow==2.15.0'],
    base_image='python:3.11'
)
def train_model(train_dataset: Input[Dataset], model_artifact: Output[Model]):
    import tensorflow as tf

 
    ds_train = tf.data.Dataset.load(train_dataset.path)
   
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
   
    model.fit(
        ds_train,
        epochs=10
    )

    model.save(model_artifact.path)

@component(
    packages_to_install=['tensorflow==2.15.0', 'numpy', 'scikit-learn'],
    base_image='python:3.11'
)
def evaluate_model(test_dataset: Input[Dataset], model_artifact: Input[Model], metrics: Output[ClassificationMetrics]):
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix

    ds_test = tf.data.Dataset.load(test_dataset.path)

    loaded_model = tf.keras.models.load_model(model_artifact.path)


    loss, accuracy = loaded_model.evaluate(ds_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    y_pred = []
    y_true = []
    for images, labels in ds_test:
        preds = loaded_model.predict(images)
       
        preds = np.argmax(preds, axis=1)
        y_pred.extend(preds)

        y_true.extend(labels.numpy())


    confusion_mtx = confusion_matrix(y_true, y_pred)
   
    metrics.log_confusion_matrix(
       [str(i) for i in range(len(confusion_mtx))],
       confusion_mtx.tolist()
    )
       
    
@component(
    packages_to_install=['tensorflow==2.15.0', 'google-cloud-storage'],
    base_image='python:3.11'
)
def tf_minio_to_gcs(tf_dataset: Input[Dataset], tf_model_artifact: Output[Model], gcs_bucket: str) -> NamedTuple('outputs', dataset_gcs_uri=str,  model_gcs_uri=str):

    import tensorflow as tf
    from google.cloud import storage
    import os
    import glob

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(gcs_bucket)

    ds = tf.data.Dataset.load(tf_dataset.path)
    
    for file_path in glob.glob(os.path.join(tf_dataset.path, "**"), recursive=True):
        print("file_path: ", file_path)
        if os.path.isfile(file_path):
            destination_path = f"{os.path.basename(tf_dataset.path)}/{os.path.basename(file_path)}"
            
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to {destination_path}")

    dataset_gcs_uri = f"gs://{gcs_bucket}/{os.path.basename(tf_dataset.path)}"
    model_gcs_uri = f"gs://{gcs_bucket}/{os.path.basename(tf_model_artifact.path)}"
    
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
