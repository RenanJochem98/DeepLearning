import pandas as pd
import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential # Modelo feedforward (a saida de uma camada eh a entrada da proxima)
from keras.layers import Dense # siginifica que todas as camadas  estao ligadas entre si
from keras.layers import Dropout # camada que zera valores aleatorios para  evitar overfiting

entradas_file_name = "datasets/breast_cancer/entradas.csv"
saidas_file_name = "datasets/breast_cancer/saidas.csv"

entradas = pd.read_csv(entradas_file_name)
saidas = pd.read_csv(saidas_file_name)

entradas_treino, entradas_teste, saidas_treino, saidas_teste = train_test_split(entradas, saidas, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dropout(0.2))
# classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=1, activation='sigmoid'))

#lr eh a taxa de learning rate
# decay eh o valor que o learning rate ira dacair durante o processo,
    #com a intencao de melhorar o desempenho do gradiente
# clipvalue eh o valor maximo que esse valor ira alcancar
otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
#eh necessario fazer a compilacao antes do treinamento
# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(entradas_treino, saidas_treino, epochs=100, batch_size=10)

previsoes = classificador.predict(entradas_teste)
previsoes = (previsoes >=0.5)

precisao = accuracy_score(saidas_teste, previsoes)
matriz = confusion_matrix(saidas_teste, previsoes)
resultado = classificador.evaluate(entradas_teste, saidas_teste)
print(precisao)
print(matriz)
print()
print(resultado)
