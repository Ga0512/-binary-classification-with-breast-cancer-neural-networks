import pandas as pd

entradas = pd.read_csv('entradas_breast.csv')
saidas = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split

entradas_treino, entradas_teste, saida_treino, saida_teste = train_test_split(entradas, saidas, test_size=0.50)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras

classificador = Sequential()
classificador.add(Dense(units=16, activation= 'relu',
                        kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=16, activation= 'relu',
                        kernel_initializer = 'random_uniform'))

classificador.add(Dense (units= 1, activation= 'sigmoid'))

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

classificador.compile(optimizer= otimizador, loss= 'binary_crossentropy',
                      metrics= ['binary_accuracy'])
classificador.fit(entradas_treino, saida_treino,
                  batch_size= 10, epochs= 100)

pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(entradas_teste)
previsoes = (previsoes > 0.7)
print(previsoes)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(saida_teste, previsoes)
print(precisao)
matriz = confusion_matrix(saida_teste, previsoes)
print(matriz)

resultado = classificador.evaluate(entradas_teste, saida_teste)
print(resultado)

print(pesos0, len(pesos0), '\n')
print('')
print(pesos1, len(pesos1), '\n')
print('')
print(pesos2, len(pesos2))
