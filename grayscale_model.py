# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import cv2
import glob
        # Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
import cv2
import glob
        # Any results you write to the current directory are saved as output.





​
from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt





DIR='/kaggle/input/mlware/data_images/train'





# Want to know how we should format the height x width image data dimensions
# for inputting to a keras model
def get_size_statistics():
    heights = []
    widths = []
    img_count = 0
    for img in os.listdir(DIR):
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            data = np.array(Image.open(path))
            heights.append(data.shape[0])
            widths.append(data.shape[1])
            img_count += 1
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))
​
get_size_statistics()





train = pd.read_csv("/kaggle/input/mlware/mnist_train_56x56_grayscale.csv")
train = train.iloc[:, 1:train.shape[1]]
train= pd.DataFrame(train).to_numpy()
train.shape





df = pd.read_csv("/kaggle/input/mlware/train.csv")
df = df.iloc[:, 1:df.shape[1]]
y = pd.DataFrame(df).to_numpy()
y.shape





X = train[:, :].reshape(train.shape[0], 1, 56, 56).astype('float32')
X = X/255.0
X.shape





X_train=X[0:4600]
X_train=np.swapaxes(X_train, 1, 3)
X_train=np.swapaxes(X_train, 1, 2)
X_test=X[4601:X.shape[0]]
X_test=np.swapaxes(X_test, 1, 3)
X_test=np.swapaxes(X_test, 1, 2)
X_train.shape





X_test.shape





y_tr=y[0:4600]
y_te=y[4601:y.shape[0]]
#y_train=np.swapaxes(y_train, 0, 1)
y_tr.shape





y_te.shape





y_train=np.zeros((y_tr.shape[0], 7))
for i in range(1,4600):
    for j in range(1,7):
        if j==y_tr[i]:
            y_train[i][j]=1 





y_test=np.zeros((y_te.shape[0], 7))
for i in range(1,3095):
    for j in range(1,7):
        if j==y_te[i]:
            y_test[i][j]=1





from PIL import Image
img=pd.DataFrame()
for filename in filenames:
    im=Image.open(os.path.join(dirname, filename))
    img=np.extend(i
​





# build model
model = Sequential()
model.add(Conv2D(32,
                 (3,3),
                 input_shape=(56, 56, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.4))
model.add(Conv2D(64,
                 (3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
#6x6x64 
#you have 5 pairs and 1 single pixel
model.add(Flatten())
​
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(7, activation='softmax'))
​
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()





model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10)
