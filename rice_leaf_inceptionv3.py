# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 00:59:32 2021

@author: chaitanya 
"""
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from keras import optimizers
adam = optimizers.Adam()

local_zip = 'C:\\Users\\chait\\Dropbox\\My PC (DESKTOP-IGK8U3R)\\Desktop\\My Folder\\ML Projects\\rice_leaf_diseases_try.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
  
base_dataset_dir = '/tmp/rice_leaf_diseases_try'
train_dir = os.path.join(base_dataset_dir, 'train')
validation_dir = os.path.join(base_dataset_dir, 'validation')
  

train_d1 = os.path.join(train_dir, 'Bacterial leaf blight')

train_d2 = os.path.join(train_dir, 'Brown spot')

train_d3 = os.path.join(train_dir, 'Leaf smut')
  

validation_d1 = os.path.join(validation_dir, 'Bacterial leaf blight')
  
validation_d2 = os.path.join(validation_dir, 'Brown spot')

validation_d3 = os.path.join(validation_dir, 'Leaf smut')


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 50,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
  
test_datagen = ImageDataGenerator( rescale = 1.0/255. )
  
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 30,
                                                    class_mode = 'categorical', 
                                                    target_size = (224, 224))     
  
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 30,
                                                          class_mode  = 'categorical', 
                                                          target_size = (224, 224))

base_model = InceptionV3(input_shape = (224, 224, 3), 
                                include_top = False, 
                                weights = 'imagenet')
for layer in base_model.layers:
  layer.trainable = False
  

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      self.model.stop_training = True

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (3, activation='softmax')(x)           
  
model = Model( base_model.input, x) 
  
model.compile(optimizer = RMSprop(lr=0.0001),loss = 'categorical_crossentropy',metrics = ['acc'])
callbacks = myCallback()
  
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 30,
            epochs = 20,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
  
epochs = range(len(acc))
  
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
  
plt.figure()
  
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
