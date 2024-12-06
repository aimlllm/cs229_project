import numpy as np
import keras
import ssl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn import metrics

ssl._create_default_https_context = ssl._create_unverified_context
train = keras.utils.image_dataset_from_directory("photoshopvsreal", labels='inferred',
    label_mode='int', image_size = (224,224), validation_split=0.1, subset='training', seed = 0)
test = keras.utils.image_dataset_from_directory("photoshopvsreal", labels='inferred',
    label_mode='int', image_size = (224,224), validation_split=0.1, subset='validation', seed = 0)


model = keras.saving.load_model('resnetmodel.keras')
convmodel = keras.saving.load_model('convresnetmodel.keras')

imagepath = ['test.jpg','test2.jpg']
weights = model.layers[-1].get_weights()[0]
weights = tf.squeeze(weights)
for imagefile in imagepath:
    image = Image.open(imagefile)
    image =  np.asarray(image)
    image = np.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        res = convmodel(image)
        grad = tape.gradient(res[1], res[0])
    #
    grad = tf.squeeze(grad)
    grad = np.mean(grad, axis = (0,1))
    res[0] = tf.squeeze(res[0])
    relu = np.inner(grad, res[0])
    relu = np.maximum(0, relu)
    plt.imshow(relu)
    plt.colorbar()
    plt.savefig('plot' + imagefile)
    plt.close()

label = np.concatenate([y for x, y in test], axis=0)
#thanks to this dude for the help with dataset labels
#https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory
#
# eval = model.evaluate(test)
pred = model.predict(test)
for i in range(len(pred)):
    if pred[i] > 0.5:
        pred[i] = 1
    else:
        pred[i] = 0
conf = metrics.confusion_matrix(label, pred)
conf = metrics.ConfusionMatrixDisplay(conf)
conf.plot()
plt.savefig('confusion.jpg')
plt.close()
