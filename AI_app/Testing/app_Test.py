import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import backend as K

batch_size=100
num_classes=10
epochs=6

filas,columnas= 28,28

(xt,yt),(xtest,ytest)= mnist.load_data()

xt=xt.reshape(xt.shape[0],filas,columnas,1)
xtest=xtest.reshape(xtest.shape[0],filas,columnas,1)

xt=xt.astype('float32')
xtest=xtest.astype('float32')

xt=xt/255
xtest=xtest/255

yt=tensorflow.keras.utils.to_categorical(yt,num_classes)
ytest=tensorflow.keras.utils.to_categorical(ytest,num_classes)

# modelo=Sequential()
# modelo.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
# modelo.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
# modelo.add(MaxPooling2D(pool_size=(2,2)))
# modelo.add(Flatten())
# modelo.add(Dense(68))
# modelo.add(Dropout(0.25))
# modelo.add(Dense(20))
# modelo.add(Dropout(0.25))
# modelo.add(Dense(num_classes,activation='softmax'))o


modelo=Sequential()
modelo.add(Flatten(input_shape=(28,28,1)))
modelo.add(Dense(68,activation='relu'))
modelo.add(Dense(20,activation='relu'))
modelo.add(Dense(num_classes,activation='softmax'))
modelo.summary()

modelo.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer=tensorflow.keras.optimizers.Adam(),metrics=['categorical_accuracy'])

modelo.fit(xt,yt,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xtest,ytest))

puntuacion=modelo.evaluate(xtest,ytest,verbose=1)

print(puntuacion)
