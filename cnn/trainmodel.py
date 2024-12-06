import numpy as np
import keras
import ssl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
# import sklearn

ssl._create_default_https_context = ssl._create_unverified_context
train = keras.utils.image_dataset_from_directory("resized_splits/train", labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))
test = keras.utils.image_dataset_from_directory("resized_splits/test", labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))
dev = keras.utils.image_dataset_from_directory("resized_splits/dev", labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
p = keras.layers.GlobalAveragePooling2D()(x)
y = keras.layers.Dense(256, activation = 'elu')(p)
f = keras.layers.Dropout(0.3)(y)
a = keras.layers.Dense(128, activation = 'elu')(f)
b = keras.layers.Dropout(0.2)(a)
outputs = keras.layers.Dense(1, activation = 'sigmoid')(b)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train, epochs=5, validation_data=test)

base_model.trainable = True
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-6,
    decay_steps=8000,
    decay_rate=0.9)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

model.fit(train, epochs=5, validation_data=test)

convmodel = keras.Model(inputs, [x, outputs])

model.evaluate(test)

model.save('xceptionCROP.keras')
convmodel.save('convxceptionCROP.keras')