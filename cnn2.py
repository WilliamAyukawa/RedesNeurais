import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Definir o número de folds para validação cruzada
k = 5

# Criar o objeto KFold
folds = KFold(n_splits=k)

# Lista para armazenar as acurácias de cada fold
acuracias = []

# Definir o callback de EarlyStopping
parada_antecipada = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Carregar o conjunto de dados MNIST
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Pré-processamento dos dados
x_treino = x_treino.reshape(-1, 28, 28, 1) / 255.0
x_teste = x_teste.reshape(-1, 28, 28, 1) / 255.0
y_treino = tf.keras.utils.to_categorical(y_treino, num_classes=10)
y_teste = tf.keras.utils.to_categorical(y_teste, num_classes=10)

# Definir a arquitetura da CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Loop sobre os folds
for treino_index, validacao_index in folds.split(x_treino):
    # Dividir os dados em fold de treinamento e fold de validação
    X_fold_treino, X_fold_validacao = x_treino[treino_index], x_treino[validacao_index]
    y_fold_treino, y_fold_validacao = y_treino[treino_index], y_treino[validacao_index]

     # Treinar o modelo no fold de treinamento
    model.fit(X_fold_treino, y_fold_treino, batch_size=64, epochs=10, callbacks=[parada_antecipada])

    # Avaliar o modelo no fold de validação
    _, acuracia = model.evaluate(X_fold_validacao, y_fold_validacao)
    acuracias.append(acuracia)

# Calcular a média das acurácias de todos os folds
media_acuracia = np.mean(acuracias)
print('Acurácia média na validação cruzada:', media_acuracia)

# Escrever os resultados em um arquivo de texto
with open('C:/estudos/RedesNeurais/RedesNeurais/resultados.txt', 'w') as file:
    file.write('Acurácia no conjunto de teste: {}\n'.format(media_acuracia))
    file.write('Histórico de treinamento:\n')
    file.write(str(history.history))