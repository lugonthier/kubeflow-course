
import logging
import argparse
import tensorflow as tf

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--is_dist', type=str, choices=["True", "False"])
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--train_dataset_path', type=str)
parser.add_argument('--model_artifact_path', type=str, default=None)
args = parser.parse_args()


lr = float(args.lr)
num_epoch = int(args.num_epoch)

is_dist = args.is_dist == "True"
num_workers = args.num_workers
batch_size_per_worker = 64
batch_size_global = batch_size_per_worker * num_workers
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.RING
    )
)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"accuracy={logs['accuracy']}")
        print(f"loss={logs['loss']}")
        logging.info(
            "Epoch {}/{}. accuracy={:.4f} - loss={:.4f}".format(
                epoch+1, num_epoch, logs["accuracy"], logs["loss"]
            )
        )
    
ds_train = tf.data.Dataset.load(args.train_dataset_path)
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    return model


if is_dist:
    logging.info("Running Distributed Training")
    logging.info("--------------------------------------------------------------------------------------\n\n")
    with strategy.scope():
        model = build_model()
else:
    logging.info("Running Single Worker Training")
    logging.info("--------------------------------------------------------------------------------------\n\n")
    model = build_model()
    

model.fit(
    ds_train,
    epochs=num_epoch,
    callbacks=[CustomCallback()],
    verbose=0
)

if args.model_artifact_path:
    model.save(args.model_artifact_path)