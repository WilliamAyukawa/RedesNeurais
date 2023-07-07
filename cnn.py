import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist

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
    tf.keras.layers.Dense(15, activation='relu'),
    #camada saida
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
historico = model.fit(x_treino, y_treino, batch_size=64, epochs=5, validation_data=(x_teste, y_teste))

# Avaliar o modelo
_, acuracia = model.evaluate(x_teste, y_teste)
print('Acurácia no conjunto de teste:', acuracia)

# Escrever os resultados em um arquivo de texto
with open('cnn.txt', 'w') as file:
    file.write('Acurácia no conjunto de teste: {}\n'.format(acuracia))
    file.write('Histórico de treinamento:\n')
    file.write(str(historico.history))
