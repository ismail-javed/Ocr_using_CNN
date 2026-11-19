import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 30
NUM_CLASSES = 26

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    label = label - 1
    return image, label

ds_train = ds_train.map(preprocess).shuffle(20000).batch(BATCH_SIZE)
ds_test  = ds_test.map(preprocess).batch(BATCH_SIZE)

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, 3, activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, 3, activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, 3, activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)

model.save("model.h5")
print("Model saved")