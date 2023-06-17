import csv
import numpy as np

# Abrir o arquivo CSV e ler os dados
with open('/home/william/Documentos/IA/portas logicas/problemAND.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)

# Converter a lista em um numpy array
data_array = np.array(data).astype(int)

# Separar as demais colunas em outro array (Entrada)
X = data_array[:, :-1]

# Separar a última coluna em um array separado (Saída esperada)
y = data_array[:, -1]

print(X.shape[1])