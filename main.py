import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam,SGD,RMSprop

PicSize = 48
path = "C:/Users/ha159/OneDrive/Documents/Facial Expression Recognition/images"
expression = 'angry'

plt.figure(figsize=(12,12))
for i in range(1,10,1):
    plt.subplot(3,3,i)
    img = load_img(path+'/train/'+expression+'/'+os.listdir(path+'/train/'+expression)[i],target_size=(PicSize,PicSize))
    plt.imshow(img)

plt.show()

batch_size = 64  # how many training files would model take in 1 iteration
data_train = ImageDataGenerator()
data_validate = ImageDataGenerator()

train_set = data_train.flow_from_directory(path+'/train/',target_size=(PicSize,PicSize),color_mode='grayscale',batch_size=batch_size,class_mode='categorical',shuffle=True)
test_set = data_train.flow_from_directory(path+'/validation/',target_size=(PicSize,PicSize),color_mode='grayscale',batch_size=batch_size,class_mode='categorical',shuffle=True)

classes = 7
model=Sequential()

model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(classes,activation='softmax'))

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# Corrected ModelCheckpoint
cp = ModelCheckpoint('./model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Other callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

cbl = [es, cp, rlr]
epochs = 48

# Corrected fit call
history = model.fit(
    train_set,
    steps_per_epoch=train_set.n // train_set.batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=test_set.n // test_set.batch_size,
    callbacks=cbl
)
model.save('model.keras')
print('hello')
