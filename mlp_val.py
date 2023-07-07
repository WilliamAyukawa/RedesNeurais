import csv
import numpy as np
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4)

# Abrir o arquivo CSV e ler os dados
with open('caracteres-Fausett/limpo_e_ruido1.csv', 'r', encoding='utf-8-sig') as file:
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
        return "Desconhecido"

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

# Dividir os dados em k-folds para validação cruzada
k = 5  # número de folds
tamanho_fold = X.shape[0] // k

# Inicializar variáveis para controle da parada antecipada
TOLERANCIA_PERDA = 50
melhor_valor_perda = np.inf
epocas_consecutivas_sem_melhora = 0

# Escrever os hiperparametros em um arquivo de texto
with open('hiperparametrosMlpVal.txt', 'w') as file:
    file.write("Hiperparametros: \n")
    file.write("    Epocas: {} \n".format(EPOCAS))
    file.write("    Taxa de aprendizado: {} \n".format(TAXA_APRENDIZADO))
    file.write("    Neurônios na camada escondida: {} \n".format(NUM_NEURONIOS_CAMADA_ESCONDIDA))
    file.write("    Quantidade de folds: {} \n".format(k))
    file.write("    Tolerancia de perda: {} \n".format(TOLERANCIA_PERDA))

# Inicializa os pesos aleatoriamente
pesos1 = np.random.random((X.shape[1], NUM_NEURONIOS_CAMADA_ESCONDIDA))
pesos2 = np.random.random((NUM_NEURONIOS_CAMADA_ESCONDIDA, Y.shape[1]))

# Escrever os pesos iniciais em um arquivo de texto
np.savetxt("pesos1IniciaisMlpVal.txt", pesos1, delimiter=',', fmt='%.3f')
np.savetxt("pesos2IniciaisMlpVal.txt", pesos2, delimiter=',', fmt='%.3f')

# Escrever os erros em um arquivo de texto
with open('errosMlpVal.txt', 'w') as file:
    # Loop de treinamento com validação cruzada
    for fold in range(k):
        # Separar os dados em fold de validação e fold de treinamento
        stop_range = ((fold + 1) * tamanho_fold) - 1
        if(stop_range > X.shape[0]):
            stop_range = X.shape[0]

        indices_validacao = range(fold * tamanho_fold, stop_range)
        indices_treinamento = np.setdiff1d(range(0, X.shape[0]), indices_validacao)

        X_treino, X_validacao = X[indices_treinamento, :], X[indices_validacao, :]
        Y_treino, Y_validacao = Y[indices_treinamento, :], Y[indices_validacao, :]

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
            
            # Calcular a perda no conjunto de treinamento
            perda_no_treinamento = np.mean(np.square(Y - layer2_output))

            # Calcular a perda no conjunto de validação
            layer1_output_val = sigmoid(np.dot(X_validacao, pesos1))
            layer2_output_val = sigmoid(np.dot(layer1_output_val, pesos2))
            valor_perda = np.mean(np.square(Y_validacao - layer2_output_val))

            # Verificar se houve melhora na perda de validação
            if valor_perda < melhor_valor_perda:
                melhor_valor_perda = valor_perda
                epocas_consecutivas_sem_melhora = 0
            else:
                epocas_consecutivas_sem_melhora += 1

            # Verificar se o treinamento deve ser interrompido
            if epocas_consecutivas_sem_melhora >= TOLERANCIA_PERDA:
                epocas_consecutivas_sem_melhora = 0
                print("Treinamento interrompido devido à falta de melhora na perda de validação. Época {}, Fold {}".format(i+1, fold+1))
                file.write("Treinamento interrompido devido à falta de melhora na perda de validação. Época {}, Fold {} \n".format(i+1, fold+1))
                break
            
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
np.savetxt("pesos1FinaisMlpVal.txt", pesos1, delimiter=',', fmt='%.3f')
np.savetxt("pesos2FinaisMlpVal.txt", pesos2, delimiter=',', fmt='%.3f')

# Testa o modelo treinado
layer1_output = sigmoid(np.dot(Xteste, pesos1))
output = sigmoid(np.dot(layer1_output, pesos2))

#Variáveis para criar matriz de confusão
array_esperado = np.array([])
array_previsto = np.array([])

# Escrever os pesos finais em um arquivo de texto
with open('saidaMlpVal.txt', 'w') as file:
    print()
    for i in range(Xteste.shape[0]):
        valorEsperado = get_letra(Yteste[i, :])
        valorPrevisto = get_letra(output[i, :])
        array_esperado = np.append(array_esperado, valorEsperado)
        array_previsto = np.append(array_previsto, valorPrevisto)
        print("Linha {}: Letra esperada[{}], Letra prevista: [{}]".format(i+1, valorEsperado, valorPrevisto))
        # Usando uma compreensão de lista para formatar os floats
        outtput_formatado = ["{:.5f}".format(num) for num in output[i, :]]
        file.write("Linha {}: Letra esperada[{}], Letra prevista: [{}], Valor Previsto: [{}] \n".format(i+1, valorEsperado, valorPrevisto, outtput_formatado))
        print()

# Calcular a matriz de confusão
matriz = confusion_matrix(array_esperado, array_previsto)

print(matriz)