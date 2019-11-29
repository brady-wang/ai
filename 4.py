import os  # 处理字符串路径
import glob  # 查找文件
from keras.models import Sequential  # 导入Sequential模型
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from pandas import Series, DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#加载数据
import os
from PIL import Image
import numpy as np
#读取文件夹train下的42000张图片，图片为彩色图，所以为3通道，
#如果是将彩色图作为输入,图像大小224*224
def load_data():
    sed = 1000
    data = np.empty((2000,224,224,3),dtype="float32")
    label = np.empty((2000,))
    imgs = os.listdir("d:/cat_dog/train/")
    num = len(imgs)
    times = 0
    time = 0
    for i in range(num):

        if imgs[i].split('.')[0] == 'cat':
            if times ==1000:
                continue
            img = Image.open("d:/cat_dog/train/" + imgs[i])

            arr = np.asarray(img, dtype="float32")
            arr.resize((224,224,3))
            data[i, :, :, :] = arr
            label[i] = 0
            times +=1


        else:

            img = Image.open("d:/cat_dog/train/" + imgs[i])

            arr = np.asarray(img, dtype="float32")
            arr.resize((224, 224, 3))
            data[1000+time, :, :, :] = arr
            label[1000+time] = 1
            time +=1
            if time == 1000:
                break

    return data,label
data,label = load_data()
print(data.shape)
train_data = data[:1800]
train_labels = label[:1800]
validation_data = data[1800:]
validation_labels = label[1800:]
