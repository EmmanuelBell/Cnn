#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:20:57 2020

@author: emmanuelmoudoute-bell
"""

# Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialiser le CNN
classifier = Sequential()

# Étape 1 - Ajouter la couche de convolution
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             input_shape=(64,64,3), 
                             activation = "relu"))

# Étape 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout d'une couche de convolution
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Étape 3 - flattening
classifier.add(Flatten())


# Étape 4 - ajout réseau de neuronne Complètement connecté
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Étape 5 - Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])


# Entrainer le CNN sur nos images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=24,
        validation_data=test_set,
        validation_steps=63,
        )

import numpy as np 
from keras.preprocessing import image

test_image = image.load_img("dataset/single_prediction/")