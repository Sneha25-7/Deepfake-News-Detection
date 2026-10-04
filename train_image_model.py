import tensorflow as tf

train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/images",
    image_size=(128,128),
    batch_size=32
)

print("Class Names:", train.class_names)
model = tf.keras.Sequential([

tf.keras.layers.Rescaling(1./255),

tf.keras.layers.Conv2D(32,3,activation="relu"),
tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Conv2D(64,3,activation="relu"),
tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Conv2D(128,3,activation="relu"),
tf.keras.layers.MaxPooling2D(),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128,activation="relu"),

tf.keras.layers.Dropout(0.4),

tf.keras.layers.Dense(1,activation="sigmoid")

])

model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
)

model.fit(train,epochs=10)

model.save("models/deepfake_model.h5")

print("Model Saved")