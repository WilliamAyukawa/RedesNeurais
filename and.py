import csv
import numpy as np

# Abrir o arquivo CSV e ler os dados
with open('/home/william/Documentos/IA/portas logicas/customAND.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)

# Converter a lista em um numpy array
data_array = np.array(data).astype(int)

# Separar as demais colunas em outro array (Entrada)
X = data_array[:, :-1]

# Separar a última coluna em um array separado (Saída esperada)
y = data_array[:, -1]

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define o número de iterações de treinamento
epochs = 10000

# Define a taxa de aprendizagem (learning rate)
learning_rate = 0.1

# Define o número de neurônios na camada oculta
hidden_neurons = 3

# Inicializa os pesos aleatoriamente
weights1 = np.random.random((X.shape[1], hidden_neurons))
weights2 = np.random.random((hidden_neurons, 1))

# Loop de treinamento
for i in range(epochs):
    # Forward pass
    layer1_output = sigmoid(np.dot(X, weights1))
    layer2_output = sigmoid(np.dot(layer1_output, weights2))
    
    # Calcula o erro na camada de saída
    layer2_error = y - layer2_output
    
    # Calcula o gradiente descendente na camada de saída
    layer2_gradient = layer2_error * sigmoid_derivative(layer2_output)
    
    # Calcula o erro na camada oculta
    layer1_error = layer2_gradient.dot(weights2.T)
    
    # Calcula o gradiente descendente na camada oculta
    layer1_gradient = layer1_error * sigmoid_derivative(layer1_output)
    
    # Atualiza os pesos
    weights2 += learning_rate * layer1_output.T.dot(layer2_gradient)
    weights1 += learning_rate * X.T.dot(layer1_gradient)

# Testa o modelo treinado
layer1_output = sigmoid(np.dot(X, weights1))
output = sigmoid(np.dot(layer1_output, weights2))

for i in range(len(X)):
    print("X = {}, y = {}, predicted y = {:.3f}".format(X[i], y[i], output[i][0]))

print()
print("X", X)
print()
print("y", y)