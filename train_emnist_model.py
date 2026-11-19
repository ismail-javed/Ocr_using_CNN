import tensorflow as tf
import tensorflow_datasets as tfds

# Load EMNIST Balanced (47 classes)
(ds_train, ds_test), info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = info.features['label'].num_classes

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
test  = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training started...")
model.fit(train, epochs=5, validation_data=test)

# Save model
model.save("model.h5")
print("model.h5 saved successfully!")
