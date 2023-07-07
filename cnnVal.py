import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Define o número de iterações de treinamento
EPOCAS = 5

# Definir o número de folds para validação cruzada
k = 5

# Criar o objeto KFold
folds = KFold(n_splits=k)

# Lista para armazenar as acurácias de cada fold
acuracias = []

# Variavel para controle da parada antecipada
TOLERANCIA_PERDA = 3

# Definir o callback de EarlyStopping
parada_antecipada = EarlyStopping(monitor='accuracy', patience=TOLERANCIA_PERDA, restore_best_weights=True)

# Define o número de neurônios na camada escondida
NUM_NEURONIOS_CAMADA_ESCONDIDA = 15

# Escrever os hiperparametros em um arquivo de texto
with open('hiperparametrosCnnVal.txt', 'w') as file:
    file.write("Hiperparametros: \n")
    file.write("    Epocas: {} \n".format(EPOCAS))
    file.write("    Neurônios na camada escondida: {} \n".format(NUM_NEURONIOS_CAMADA_ESCONDIDA))
    file.write("    Quantidade de folds: {} \n".format(k))
    file.write("    Tolerancia de perda: {} \n".format(TOLERANCIA_PERDA))

# Carregar o conjunto de dados MNIST
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Pré-processamento dos dados
x_treino = x_treino.reshape(-1, 28, 28, 1) / 255.0
x_teste = x_teste.reshape(-1, 28, 28, 1) / 255.0
y_treino = tf.keras.utils.to_categorical(y_treino, num_classes=10)
y_teste = tf.keras.utils.to_categorical(y_teste, num_classes=10)

# Definir a arquitetura da CNN
model = tf.keras.Sequential([
    #camada de entrada
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    #camada escondida
    tf.keras.layers.Dense(NUM_NEURONIOS_CAMADA_ESCONDIDA, activation='relu'),
    #camada saida
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Obter os pesos iniciais
pesos_iniciais = []
for layer in model.layers:
    pesos_iniciais.append(layer.get_weights())

# Escrever os resultados em um arquivo de texto
with open('cnnVal.txt', 'w') as file:
    # Loop sobre os folds
    for treino_index, validacao_index in folds.split(x_treino):
        # Dividir os dados em fold de treinamento e fold de validação
        X_fold_treino, X_fold_validacao = x_treino[treino_index], x_treino[validacao_index]
        y_fold_treino, y_fold_validacao = y_treino[treino_index], y_treino[validacao_index]

        # Treinar o modelo no fold de treinamento
        historico = model.fit(X_fold_treino, y_fold_treino, batch_size=64, epochs=EPOCAS, callbacks=[parada_antecipada])
        file.write('Treinamento iteracao {}:\n'.format(str(historico.history)))

        # Avaliar o modelo no fold de validação
        _, acuracia = model.evaluate(X_fold_validacao, y_fold_validacao)
        acuracias.append(acuracia)

    # Calcular a média das acurácias de todos os folds
    media_acuracia = np.mean(acuracias)
    print('Acurácia média na validação cruzada:', media_acuracia)

    # Obter os pesos finais
    pesos_finais = []
    for layer in model.layers:
        pesos_finais.append(layer.get_weights())

    # Imprimir os pesos iniciais e finais de cada camada
    for i in range(len(pesos_finais)):
        print("Camada {}: pesos iniciais = {}, pesos finais = {}".format(
            i+1, pesos_iniciais[i], pesos_finais[i]))

    file.write('Acurácia no conjunto de teste: {}\n'.format(media_acuracia))
