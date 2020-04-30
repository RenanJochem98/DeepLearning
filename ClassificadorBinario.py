import pandas as pd
import numpy as np
import keras

from keras.models import Sequential # Modelo feedforward (a saida de uma camada eh a entrada da proxima)
from keras.layers import Dense # siginifica que todas as camadas  estao ligadas entre si
from keras.layers import Dropout # camada que zera valores aleatorios para  evitar overfiting

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class ClassificadorBinario():

    def __init__(self, entradas_file_name, saidas_file_name):
        self.load_data(entradas_file_name, saidas_file_name)
        self.classificador = Sequential()
        self.otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

    def load_data(self, entradas_file_name, saidas_file_name):
        self.entradas = pd.read_csv(entradas_file_name)
        self.saidas = pd.read_csv(saidas_file_name)

        self.entradas_treino, self.entradas_teste, self.saidas_treino, self.saidas_teste = train_test_split(self.entradas, self.saidas, test_size=0.25)

    def create_network(self, num_layers):
        for i in range(1,num_layers):
            self.classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
            self.classificador.add(Dropout(0.2))

        self.classificador.add(Dense(units=1, activation='sigmoid'))
        self.classificador.compile(optimizer=self.otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

    def create_trained_network(self, num_layers = 1):
        self.create_network()
        self.classificador.fit(self.entradas_treino, self.saidas_treino, epochs=100, batch_size=10)
        return self.classificador

    def predict(self):
        previsoes = self.classificador.predict(self.entradas_teste)
        return previsoes

    def show_results(self, previsoes):
        previsoes = (previsoes >=0.5)

        precisao = accuracy_score(saidas_teste, previsoes)
        matriz = confusion_matrix(saidas_teste, previsoes)
        resultado = self.classificador.evaluate(self.entradas_teste, self.saidas_teste)
        print(precisao)
        print(matriz)
        print()
        print(resultado)
