import numpy as np
import cv2
import os
import pandas as pd
from six.moves import cPickle

# For this problem the validation and test data provided by the concerned authority did not have labels, so the training data was split into train, test and validation sets
train_dir = './data/boneage-training-dataset/'

X_train = []
y_age = []
y_gender = []

df = pd.read_csv('./data/boneage-training-dataset.csv')
a = df.to_numpy()
m = a.shape[0]

j = 0
path = train_dir
k = 0

print('Loading data set...')
for i in os.listdir(path):
    y_age.append(int([value.boneage for (key, value) in df[df.id == int(i[:-4])].iterrows()][0]))
    print(y_age)
    a = [value.male for key, value in df[df.id == int(i[:-4])].iterrows()][0]
    if a:
        y_gender.append(1)
    else:
        y_gender.append(0)
    img_path = path + i
    img = cv2.imread(img_path)
    print(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    x = np.asarray(img, dtype=np.uint8)
    X_train.append(x)

print('100% completed loading data')

# Save data
train_pkl = open('data1.pkl', 'wb')
cPickle.dump(X_train, train_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_pkl.close()

train_age_pkl = open('data_age.pkl', 'wb')
cPickle.dump(y_age, train_age_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_age_pkl.close()

train_gender_pkl = open('data_gender.pkl', 'wb')
cPickle.dump(y_gender, train_gender_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_gender_pkl.close()