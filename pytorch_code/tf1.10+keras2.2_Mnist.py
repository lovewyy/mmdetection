# tensorflow_2.0 can't run, need 1.0
import tensorflow as tf

# import tensorflow.compat.v1 as tf

print(tf.__version__)
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

print(keras.__version__)
sess = tf.InteractiveSession()

# 连接不上
from tensorflow.examples.tutorials.mnist import input_data
#路径要写全
MNIST_data_folder="D:\\Users\\l50002801\\PycharmProjects\\HelloWorld\\MNIST"
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)

X_train,Y_train = mnist.train.images,mnist.train.labels
X_test,Y_test = mnist.test.images,mnist.test.labels
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model = Sequential()
# 第一个卷积层（后接池化层）：
model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# 第二个卷积层（后接池化层）：
model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# 将上面的结果扁平化，然后接全连接层：
model.add(Flatten())
model.add(Dense(128,activation='relu'))

#最后一个Softmax输出：
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

X_train_image = X_train.reshape(X_train.shape[0],28,28,1)
X_test_image = X_test.reshape(X_test.shape[0],28,28,1)
# 开始训练：
model.fit(X_train_image,Y_train,epochs=6,batch_size=64)
