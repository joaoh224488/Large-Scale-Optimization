import sys
import logging
import highspy

from highspy import Highs

class HighsSolver:
    # Inicializa o solver HiGHS e define o caminho do arquivo MPS
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.model = Highs()
        self.res = None

    # Recebe o caminho para o arquivo MPS, lê o arquivo, executa o solver e armazena o resultado
    def run(self):
        try:
            # Carregar o modelo a partir do arquivo MPS
            status = self.model.readModel(self.instance_path)
            
            if status != highspy.HighsStatus.kOk:
                raise Exception("Erro ao carregar o modelo MPS.")
            
            # Resolver o problema de otimização
            self.model.run()
            self.res = self.model.getSolution()
        
        except Exception as e:
            logging.error(f"Erro na execução do solver: {e}")
            self.res = None

    # Imprime no console os resultados do problema
    def print_results(self):
        if self.res is None:
            print("Nenhum resultado disponível.")
            return
        
        try:
            print(f"Status: {self.model.modelStatusToString(self.model.getModelStatus())}")
            print(f"Valor objetivo: {self.model.getObjectiveValue()}")
            print(f"success: {self.model.getModelStatus()}")
            print(f"Número de iterações: {self.model.getInfo().simplex_iteration_count}")
        
        except Exception as e:
            raise Exception(f"Erro ao imprimir resultados: {e}")
    
    # Retorna um dict com os resultados do problema
    def get_results(self):
        if self.res is None:
            return None
        
        try:
            return {
                "status": self.model.modelStatusToString(self.model.getModelStatus()),
                "objective_value": self.model.getObjectiveValue(),
                "success": self.model.getModelStatus(),
                "iterations": self.model.getInfo().simplex_iteration_count,
            }
        
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            return None
