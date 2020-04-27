import pandas as pd
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense # siginifica que todas as camadas  estao ligadas entre si

entradas_file_name = "datasets/breast_cancer/entradas.csv"
saidas_file_name = "datasets/breast_cancer/saidas.csv"

entradas = pd.read_csv(entradas_file_name)
saidas = pd.read_csv(saidas_file_name)

entradas_treino, entradas_teste, saidas_treino, saidas_teste = train_test_split(entradas, saidas, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=1, activation='sigmoid'))

#eh necessario fazer a compilacao antes do treinamento
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(entradas_treino, saidas_treino, epochs=1000)
