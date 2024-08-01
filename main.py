import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

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
print('hello')