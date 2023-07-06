import os
import csv
import numpy as np

# Abrir o arquivo CSV e ler os dados
with open('caracteres-Fausett/limpo_e_ruido1.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)

# Converter a lista em um numpy array
data_array = np.array(data).astype(int)

with open('caracteres-Fausett/customRuido2.csv', 'r', encoding='utf-8-sig') as file:
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
        if val < 0.4 or val > 0.6:
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
EPOCAS = 20000

# Define a taxa de aprendizagem
TAXA_APRENDIZADO = 0.1

# Define o número de neurônios na camada escondida
NUM_NEURONIOS_CAMADA_ESCONDIDA = 18

# Inicializa os pesos aleatoriamente
pesos1 = np.random.random((X.shape[1], NUM_NEURONIOS_CAMADA_ESCONDIDA))
pesos2 = np.random.random((NUM_NEURONIOS_CAMADA_ESCONDIDA, Y.shape[1]))

# Escrever os hiperparametros em um arquivo de texto
with open('hiperparametrosMlp.txt', 'w') as file:
    file.write("Hiperparametros: \n")
    file.write("    Epocas: {} \n".format(EPOCAS))
    file.write("    Taxa de aprendizado: {} \n".format(TAXA_APRENDIZADO))
    file.write("    Neurônios na camada escondida: {} \n".format(NUM_NEURONIOS_CAMADA_ESCONDIDA))


# Escrever os pesos iniciais em um arquivo de texto
np.savetxt("pesos1IniciaisMlp.txt", pesos1, delimiter=',', fmt='%.3f')
np.savetxt("pesos2IniciaisMlp.txt", pesos2, delimiter=',', fmt='%.3f')

# Escrever os erros em um arquivo de texto
with open('errosMlp.txt', 'w') as file:
    # Loop de treinamento
    for i in range(EPOCAS):

        if(i % 1000 == 0):
            print("Iteracao {}".format(i))

        # Forward pass
        layer1_output = sigmoid(np.dot(X, pesos1))
        layer2_output = sigmoid(np.dot(layer1_output, pesos2))
        
        # Calcula o erro na camada de saída
        layer2_error = Y - layer2_output

        file.write("Iteracao {} -> Valor erro: {} \n".format(i, layer2_error))
        
        # Calcula o gradiente descendente na camada de saída
        layer2_gradient = layer2_error * sigmoid_derivative(layer2_output)
        
        # Calcula o erro na camada oculta
        layer1_error = layer2_gradient.dot(pesos2.T)
        
        # Calcula o gradiente descendente na camada oculta
        layer1_gradient = layer1_error * sigmoid_derivative(layer1_output)
        
        # Atualiza os pesos
        pesos2 += TAXA_APRENDIZADO * layer1_output.T.dot(layer2_gradient)
        pesos1 += TAXA_APRENDIZADO * X.T.dot(layer1_gradient)

# Escrever os pesos finais em um arquivo de texto
np.savetxt("pesos1FinaisMlp.txt", pesos1, delimiter=',', fmt='%.3f')
np.savetxt("pesos2FinaisMlp.txt", pesos2, delimiter=',', fmt='%.3f')

# Testa o modelo treinado
layer1_output = sigmoid(np.dot(Xteste, pesos1))
output = sigmoid(np.dot(layer1_output, pesos2))

# Escrever os pesos finais em um arquivo de texto
with open('saidaMlp.txt', 'w') as file:
    print()
    for i in range(Xteste.shape[0]):
        valorEsperado = get_letra(Yteste[i, :])
        valorPrevisto = get_letra(output[i, :])
        print("Linha {}: Valor esperado[{}], Valor previsto: [{}], Correto?{} | {}".format(i+1, valorEsperado, valorPrevisto, (valorEsperado==valorPrevisto), output[i, :]))
        file.write("Linha {}: Valor esperado[{}], Valor previsto: [{}], Correto?{} | {} \n".format(i+1, valorEsperado, valorPrevisto, (valorEsperado==valorPrevisto), output[i, :]))
        print()