import tensorflow as tf
import numpy as np

# Definir os dados de entrada e saída
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Definir a arquitetura da MLP
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Treinar o modelo
model.fit(X, y, epochs=10000)

# Fazer previsões
predictions = model.predict(X)
rounded_predictions = np.round(predictions, 3)
print(rounded_predictions)
