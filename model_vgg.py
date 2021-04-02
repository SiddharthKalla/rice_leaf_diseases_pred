# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:34:47 2021

@author: leno
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_SIZE = [224, 224]

#Give dataset path
train_path = 'C:\\Users\\leno\\Desktop\\Machine Learning Stuff\\Research_paper_1\\data1\\train'
test_path = 'C:\\Users\\leno\\Desktop\\Machine Learning Stuff\\Research_paper_1\\data1\\test'

from PIL import Image 
import os 
from IPython.display import display
from IPython.display import Image as _Imgdis
# creating a object  

  
folder = train_path+'/Brown spot'


onlybenignfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} images".format(len(onlybenignfiles)))
print("Image examples: ")


for i in range(10):
    print(onlybenignfiles[i])
    display(_Imgdis(filename=folder + "/" + onlybenignfiles[i], width=240, height=240))
    
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
vgg.input
for layer in vgg.layers:
  layer.trainable = False
  
folders = glob('C:\\Users\\leno\\Desktop\\Machine Learning Stuff\\Research_paper_1\\data1\\train\\*')
print(len(folders))
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

from keras import optimizers


adam = optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

from datetime import datetime
from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                               verbose=2, save_best_only=True)

callbacks = [checkpoint]

start = datetime.now()

model_history=model.fit_generator(
  train_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=5,
  validation_steps=32,
  callbacks = callbacks, verbose = 2)


duration = datetime.now() - start
print("Training completed in time: ", duration)

_# Plot training & validation loss values
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('CNN Model accuracy values')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

##prediction_func
'''import cv2
import tensorflow as tf

categories = ["Bacterial leaf blight","Brown spot","Leaf smut"]

def prepare(filepath):
  IMG_SIZE = 224
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("mymodel.h5")

prediction = model.predict([prepare('dog.4004.jpg')])

image = cv2.imread('DSC_0513.jpg')
image = np.expand_dims(image, axis=0)
img_array = cv2.imread('DSC_0513.jpg', cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (224,224))
new_array.reshape(-1, 224, 224, 1)

image = cv2.reshape(image, (-1,244,244,3))

img_array = cv2.imread('DSC_0513.jpg', cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (240,240))
new_array.reshape(-1, 240,240, 1)'''
