import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento dos dados
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Definir a arquitetura da CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Avaliar o modelo
_, accuracy = model.evaluate(x_test, y_test)
print('Acurácia no conjunto de teste:', accuracy)

# Escrever os resultados em um arquivo de texto
with open('resultados.txt', 'w') as file:
    file.write('Acurácia no conjunto de teste: {}\n'.format(accuracy))
    file.write('Histórico de treinamento:\n')
    file.write(str(history.history))
