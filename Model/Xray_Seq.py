#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
from skimage.io import imread, imsave,imshow
from skimage.transform import resize
from skimage import exposure
import os
FAST_RUN = False
IMAGE_WIDTH=256
IMAGE_HEIGHT=256
IMAGE_CHANNELS=1 
droput_rate = 0.1
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
epochs= 800
print(os.listdir("."))
df = pd.read_csv('./sample_labels.csv', usecols=["Image Index","Finding Labels"])


# In[ ]:


df.loc[df['Finding Labels'] == 'No Finding', 'Finding Labels'] = 'NF'
df.loc[df['Finding Labels'] != 'NF', 'Finding Labels'] = 'F'



print(os.listdir("."))



train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



train_df.shape



validate_df







# RGB color
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droput_rate))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droput_rate))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droput_rate))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droput_rate))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(droput_rate))
model.add(Dense(1, activation='sigmoid'))
adam_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer= adam_optim, metrics=['accuracy'])

model.summary()


# In[123]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[124]:


earlystop = EarlyStopping(patience=10)


# In[125]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[126]:


callbacks2 = [earlystop, learning_rate_reduction]


# In[127]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=64


# In[128]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


# In[135]:


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./sample/images/", 
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=IMAGE_SIZE,
    color_mode = 'grayscale',
  class_mode='binary',
    batch_size=batch_size
)


# In[136]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./sample/images/", 
    x_col='Image Index',
    y_col='Finding Labels',
    target_size=IMAGE_SIZE,
    color_mode = 'grayscale',
    class_mode='binary',
    batch_size=batch_size
)



print (train_df.shape)




#3 if FAST_RUN else 100
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks2
)
model.save('test_model.h5')
training_loss = history.history['loss']
training_accuracy = history.history['acc']

validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_acc']
df = pd.DataFrame(data={'training loss': training_loss,'validation loss': validation_loss, 'training accuracy':training_accuracy,'validation accuracy':validation_accuracy})

df.to_csv('losses2.csv')


# In[ ]:




