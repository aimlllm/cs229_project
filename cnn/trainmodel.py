import numpy as np
import keras
import ssl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn import metrics
from sklearn import preprocessing
import os
import math

dirTrain = 'Casia_Cropped/train'
dirTest = 'Casia_Cropped/test'
dirVal = 'Casia_Cropped/dev'
modelname = 'Xception_Casia_Cropped'

dir = ["Casia_Cropped/dev/Au", "Casia_Cropped/dev/Tp"]
dest = 'Trained_Xception_Casia_Cropped_L1/'

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
    image_size=(224, 224), shuffle = False)

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
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train, epochs = 20, validation_data=test)

# training whole model

base_model.trainable = True
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])

model.fit(train, epochs=10, validation_data=test)

model.evaluate(test)
#saving subset of model for accuracy.py
convmodel = keras.Model(inputs, [x, outputs])


label = np.concatenate([y for x, y in val], axis=0)
#thanks to this dude for the help with dataset labels
#https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory
#
# eval = model.evaluate(test)
pred = model.predict(val)
for i in range(len(pred)):
    if pred[i] > 0.5:
        pred[i] = 1
    else:
        pred[i] = 0
conf = metrics.confusion_matrix(label, pred)
conf = metrics.ConfusionMatrixDisplay(conf)
conf.plot()
plt.savefig(dest + 'confusion_' + modelname + '.jpg')
plt.close()

index = 0
weights = model.layers[-1].get_weights()[0]
weights = tf.squeeze(weights)


for i in range(2):
    for filename in os.listdir(dir[i]):
        f = os.path.join(dir[i], filename)
        if os.path.isfile(f) and filename.endswith('.jpg'):
            image = Image.open(f)
            image = np.asarray(image)
            image = np.expand_dims(image, axis=0)

            with tf.GradientTape() as tape:
                res = convmodel(image)
                grad = tape.gradient(res[1], res[0])

            grad = tf.squeeze(grad)
            grad = np.mean(grad, axis=(0, 1))
            res[0] = tf.squeeze(res[0])
            relu = np.inner(grad, res[0])
            relu = np.maximum(0, relu)
            image = tf.squeeze(image)
            relu = preprocessing.normalize(relu)

            relu = Image.fromarray(np.uint8(255* relu))
            relu = relu.resize((224, 224))

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            ax.imshow(relu, cmap='jet', alpha=0.5)
            plt.axis('off')

            if pred[index] == 0 and label[index] == 0:
                name = "tn_" + filename
            elif pred[index] == 1 and label[index] == 1:
                name = "tp_" + filename
            elif pred[index] == 1 and label[index] == 0:
                name = "fp_" + filename
            else:
                name = "fn_" + filename

            plt.savefig(dest + name, format = 'jpg')
            plt.close()
            index +=1

model.save(dest + modelname + '.keras')
convmodel.save(dest + 'conv' + modelname + '.keras')