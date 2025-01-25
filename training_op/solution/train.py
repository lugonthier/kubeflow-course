import os
import glob
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
from google.cloud import storage


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--train_dataset_path', type=str, default=None)
parser.add_argument('--model_artifact_path', type=str, default=None)
args = parser.parse_args()

if args.train_dataset_path:
    ds = tf.data.Dataset.load(args.train_dataset_path)
else:
    (ds, _), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    ds = ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.batch(64)
    ds = ds.prefetch(tf.data.AUTOTUNE)  
    
    

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    run_eagerly=True
)

model.fit(
    ds,
    epochs=args.num_epoch
)

if args.model_artifact_path:
    
    bucket_name, path = args.model_artifact_path.split('/')[2], '/'.join(args.model_artifact_path.split('/')[3:])
    model.save('/temp/model')
    
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
        
    for file_path in glob.glob(os.path.join('/temp/model', "**"), recursive=True):
        if os.path.isfile(file_path):
            blob = bucket.blob(f"{path}/{os.path.basename(file_path)}")
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to {os.path.basename(file_path)}")