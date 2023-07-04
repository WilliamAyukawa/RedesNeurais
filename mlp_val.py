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

# Dividir os dados em k-folds para validação cruzada
k = 5  # número de folds
tamanho_fold = len(X) // k

# Inicializar variáveis para controle da parada antecipada
TOLERANCIA_PERDA = 5
melhor_valor_perda = np.inf
epocas_consecutivas_sem_melhora = 1

# Inicializa os pesos aleatoriamente
pesos1 = np.random.random((X.shape[1], NUM_NEURONIOS_CAMADA_ESCONDIDA))
pesos2 = np.random.random((NUM_NEURONIOS_CAMADA_ESCONDIDA, Y.shape[1]))

# Loop de treinamento com validação cruzada
for fold in range(k):
    # Separar os dados em fold de validação e fold de treinamento
    valor_indices = range(fold * tamanho_fold, (fold + 1) * tamanho_fold)
    indices_treinamento = [i for i in range(X.shape[0]) if i not in tamanho_fold]

    X_treino, X_validacao = X[indices_treinamento], X[valor_indices]
    Y_treino, Y_validacao = Y[indices_treinamento], Y[valor_indices]
    # Loop de treinamento
    for i in range(EPOCAS):
        # Forward pass
        layer1_output = sigmoid(np.dot(X, pesos1))
        layer2_output = sigmoid(np.dot(layer1_output, pesos2))

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
            print("Treinamento interrompido devido à falta de melhora na perda de validação.")
            break
        
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