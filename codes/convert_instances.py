
import os
from highspy import Highs

# Pastas de entrada e saída
entrada = 'C:/Users/ismae/Desktop/Atividades/Large-Scale-Optimization/Instancias/mps'
saida = 'C:/Users/ismae/Desktop/Atividades/Large-Scale-Optimization/Instancias/lp'

# Criar pasta de saída se não existir
os.makedirs(saida, exist_ok=True)

# Listar e converter todos os arquivos .mps
for nome_arquivo in os.listdir(entrada):
    if nome_arquivo.endswith('.mps'):
        caminho_mps = os.path.join(entrada, nome_arquivo)
        caminho_mps = caminho_mps.replace("\\", "/")
        
        highs = Highs()
        highs.readModel(caminho_mps)
        
        nome_lp = os.path.splitext(nome_arquivo)[0] + '.lp'
        caminho_lp = os.path.join(saida, nome_lp)
        caminho_lp = caminho_lp.replace("\\", "/")
        highs.writeModel(caminho_lp)
        
        print(f'Convertido: {nome_arquivo} -> {nome_lp}')
