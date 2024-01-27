from keras.preprocessing.image import ImageDataGenerator
import cv2
import pickle
import numpy as np


datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='constant',
    cval=0
)

with open('data1.pkl', 'rb') as f:
    img_arr = np.array(pickle.load(f))


x = x.reshape((1,)+x.shape)

i = 0
for batch in datagen.flow(x, batch_size=10,
                          save_to_dir='augmented',
                          save_prefix='aug',
                          save_format='png'
                         ):
    i+=1
    if i ==20:
        break