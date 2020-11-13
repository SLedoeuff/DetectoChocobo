#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Créer par Sacha Le Doeuff

from keras import losses
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation,Conv2D,Flatten
from sklearn.model_selection import train_test_split

from time import time
from PIL import Image
from resizeimage import resizeimage
import pandas as pd
import numpy as np
import glob
import random

"""-------------------------------------------------------
#        Resize des images + conversion en vecteur       # 
-------------------------------------------------------"""

originalShape = []
ImgArray = []
ImgType = []
#matrixImgArrayById = [ ImgType, ImgArray ]

filesOui = glob.glob('OuiChoco/*.jpg')
filesNon = glob.glob('NonChoco/*.jpg')

#image de chocobo
for file in filesOui:
	with open(file, 'r+b') as f:
		with Image.open(f) as image:
			#img = Image.open(file)
			data = np.asarray(image, dtype="int32")

			# record the original shape
			originalShape.append(data)

			# make a 1-dimensional view of arr, on aurai pu faire un reshape
			flat_arr = data.ravel()
			ImgArray.append(flat_arr)
			ImgType.append(1)

#image d'autre chose que des chocobo
for file in filesNon:
	with open(file, 'r+b') as f:
		with Image.open(f) as image:
			#img = Image.open(file)
			data = np.asarray(image, dtype="int32")

			# record the original shape
			originalShape.append(data)

			# make a 1-dimensional view of arr, on aurai pu faire un reshape
			flat_arr = data.ravel()
			ImgArray.append(flat_arr)
			ImgType.append(0)


#random.shuffle(matrixImgArrayById)


"""-------------------------------------------------------
#                    Création du CNN                     # 
-------------------------------------------------------"""

checkpoint = ModelCheckpoint("bestChocoboModel.h5", monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

Y = np.asarray(ImgType)
X = np.asarray(originalShape)

#dataset
x_train, x_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5)

#modele
model = Sequential()
model.add(Conv2D(92, (3 , 3), input_shape=(128,128,3), activation='relu'))
model.add(Conv2D(76, (3 , 3), activation='relu'))
model.add(Conv2D(64, (3 , 3), activation='selu'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])

#model.fit(x_train, y_train, nb_epoch=5, steps_per_epoch=10, validation_steps=5, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpoint])
model.fit(x_train, y_train, nb_epoch=10, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpoint])
score = model.evaluate(x_test, y_test, verbose=0)

print('Meilleur Modele :')

# meilleur modele enregistré

model2 = load_model("bestChocoboModel.h5")
score2 = model2.evaluate(x_test, y_test, verbose=0)
print('Test score:', score2[0])
print('Test accuracy:', score2[1])

print(model.predict(np.array(x_test)))
