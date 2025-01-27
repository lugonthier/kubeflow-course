def train_func(params):

    import tensorflow as tf
    import tensorflow_datasets as tfds

    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(64)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params['lr']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=params['num_epoch']
    )
 
    


if __name__=="__main__":
    import argparse
    from kubeflow.training import TrainingClient
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--namespace", type=str, default="team-1")
    parser.add_argument("--training_job_name", type=str)
    parser.add_argument(
        "--training_job_image", type=str, default="tensorflow/tensorflow:2.15.0"
    )

    
    args = parser.parse_args()
    
    training_client = TrainingClient()
    training_client.create_tfjob_from_func(
        name=args.training_job_name,
        func=train_func,
        parameters={
                    'lr': args.learning_rate, 'num_epoch': args.num_epoch},
        num_worker_replicas=1,
        namespace=args.namespace,
        base_image=args.training_job_image,
        packages_to_install=["tensorflow-datasets"]
    )

    training_client.wait_for_job_conditions(name=args.training_job_name, namespace=args.namespace)


    training_client.get_job_logs(name=args.training_job_name, namespace=args.namespace)