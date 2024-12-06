import numpy as np
import keras
import ssl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
# import sklearn

dirTrain = 'CASIA_Cropped/train'
dirTest = 'CASIA_Cropped/test'
dirVal = 'CASIA_Cropped/dev'
modelname = 'Xception_Casia_Cropped'

#load datasets
ssl._create_default_https_context = ssl._create_unverified_context
train = keras.utils.image_dataset_from_directory(dirTrain, labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))
test = keras.utils.image_dataset_from_directory(dirTest, labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))
val = keras.utils.image_dataset_from_directory(dirVal, labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224))

#import model
base_model = keras.applications.Xception(
    weights= 'imagenet',
    input_shape=(224, 224, 3),
    include_top=False)

#initializing and training top layers
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
p = keras.layers.GlobalAveragePooling2D()(x)
y = keras.layers.Dense(256, activation = 'relu')(p)
f = keras.layers.Dropout(0.4)(y)
a = keras.layers.Dense(128, activation = 'relu')(f)
b = keras.layers.Dropout(0.4)(a)
outputs = keras.layers.Dense(1, activation = 'sigmoid')(b)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train, epochs=10, validation_data=test)

#training whole model
base_model.trainable = True
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=8000,
    decay_rate=0.9)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

model.fit(train, epochs=20, validation_data=test)

#saving subset of model for accuracy.py
convmodel = keras.Model(inputs, [x, outputs])
#
# print("EVALUATING ON DEV")
# model.evaluate(val)
# pred = model.predict(val)
# print(pred)

model.save(modelname + '.keras')
convmodel.save('conv_' + modelname + '.keras')