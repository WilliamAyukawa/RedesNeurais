import csv
import numpy as np

# Abrir o arquivo CSV e ler os dados
with open('caracteres-Fausett/customLimpo.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)

# Converter a lista em um numpy array
data_array = np.array(data).astype(int)

with open('caracteres-Fausett/customRuido.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)

teste_array = np.array(data).astype(int)

def get_letra(array):
    """
    Converte o vetor em letra
    """
    # Verificar a proximidade de cada valor com 0 ou 1
    entrada = [round(val, 3) for val in array]
    linha = []

    for val in entrada:
        if val < 0.2 or val > 0.8:
            linha.append(round(val))
        else:
            linha.append(val)

    # Classificar em letras de acordo com os valores do vetor
    if linha == [1, 0, 0, 0, 0, 0, 0]:
        return "A"
    elif linha == [0, 1, 0, 0, 0, 0, 0]:
        return "B"
    elif linha == [0, 0, 1, 0, 0, 0, 0]:
        return "C"
    elif linha == [0, 0, 0, 1, 0, 0, 0]:
        return "D"
    elif linha == [0, 0, 0, 0, 1, 0, 0]:
        return "E"
    elif linha == [0, 0, 0, 0, 0, 1, 0]:
        return "J"
    elif linha == [0, 0, 0, 0, 0, 0, 1]:
        return "K"
    else:
        return "Unknown {}".format(linha)

# Separar as primeiras 63 colunas em um array (Entrada)
X = data_array[:, :63]

# Separar as 7 últimas colunas em um array separado (Saída esperada)
Y = data_array[:, 63:]

Xteste = teste_array[:, :63]
Yteste = teste_array[:, 63:]

def sigmoid(x):
    """
    Define a função de ativacao sigmoide
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Define a função de derivada da sigmoide
    """
    return x * (1 - x)

# Define o número de iterações de treinamento
EPOCAS = 30000

# Define a taxa de aprendizagem
TAXA_APRENDIZADO = 0.2

# Define o número de neurônios na camada escondida
NUM_NEURONIOS_CAMADA_ESCONDIDA = 12

# Inicializa os pesos aleatoriamente
pesos1 = np.random.random((X.shape[1], NUM_NEURONIOS_CAMADA_ESCONDIDA))
pesos2 = np.random.random((NUM_NEURONIOS_CAMADA_ESCONDIDA, Y.shape[1]))

# Loop de treinamento
for i in range(EPOCAS):
    # Forward pass
    layer1_output = sigmoid(np.dot(X, pesos1))
    layer2_output = sigmoid(np.dot(layer1_output, pesos2))
    
    # Calcula o erro na camada de saída
    layer2_error = Y - layer2_output
    
    # Calcula o gradiente descendente na camada de saída
    layer2_gradient = layer2_error * sigmoid_derivative(layer2_output)
    
    # Calcula o erro na camada oculta
    layer1_error = layer2_gradient.dot(pesos2.T)
    
    # Calcula o gradiente descendente na camada oculta
    layer1_gradient = layer1_error * sigmoid_derivative(layer1_output)
    
    # Atualiza os pesos
    pesos2 += TAXA_APRENDIZADO * layer1_output.T.dot(layer2_gradient)
    pesos1 += TAXA_APRENDIZADO * X.T.dot(layer1_gradient)

# Testa o modelo treinado
layer1_output = sigmoid(np.dot(X, pesos1))
output = sigmoid(np.dot(layer1_output, pesos2))


layer1_output = sigmoid(np.dot(Xteste, pesos1))
output = sigmoid(np.dot(layer1_output, pesos2))

print()
for i in range(Xteste.shape[0]):
    print("Linha {}: Valor esperado[{}], Valor previsto: [{}]".format(i+1, get_letra(Yteste[i, :]), get_letra(output[i, :])))
    print()