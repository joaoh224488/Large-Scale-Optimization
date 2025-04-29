import sys
import logging
import highspy as hs
import numpy as np

from time import perf_counter as pc
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
        """
        Imprime os resultados da otimização no console.

        Exibe:
        - Status do modelo
        - Valor da função objetivo
        - Indicador de sucesso
        - Número de iterações do simplex

        Se não houver resultado (self.res é None) ou ocorrer erro:
        - Imprime mensagem apropriada
        - Levanta exceção em caso de erro na impressão
        """
		
        if self.res is None:
            print("Nenhum resultado disponível.")
            return
        
        try:
            results = self.get_results()
            for key, value in results.items():
                print(f"{key}: {value}")
        
        except Exception as e:
            raise Exception(f"Erro ao imprimir resultados: {e}")
    
    # Retorna um dict com os resultados do problema
    def get_results(self):
        """
        Retorna os resultados da otimização em formato de dicionário.

        Returns:
            dict: Dicionário contendo:
                - status: Status do modelo em formato string
                - objective_value: Valor final da função objetivo
                - success: Status numérico do modelo
                - iterations: Número de iterações do simplex
            None: Se não houver resultado ou ocorrer erro

        O método captura exceções e registra erros no log caso ocorram.
        """
        if self.res is None:
            return None
        
        try:
            status = self.model.getModelStatus()
            status_str = self.model.modelStatusToString(status)
            primal_value = self.model.getObjectiveValue() if status == hs.HighsModelStatus.kOptimal else None
            
            # Cálculo do valor dual e gap
            dual_value = None
            gap = None
            if status == hs.HighsModelStatus.kOptimal:
                solution = self.model.getSolution()
                dual_vars = solution.row_dual
                
                lp_model = self.model.getLp()
                row_lower = lp_model.row_lower_
                row_upper = lp_model.row_upper_
                num_rows = len(row_lower)
                
                b = []
                for i in range(num_rows):
                    if row_lower[i] == row_upper[i]:
                        b.append(row_lower[i])
                    elif row_upper[i] < hs.kHighsInf:
                        b.append(row_upper[i])
                    else:
                        b.append(row_lower[i])
                
                dual_value = np.dot(b, dual_vars)
                gap = primal_value - dual_value if (primal_value is not None and dual_value is not None) else None
            
            # Métricas de inviabilidade
            info = self.model.getInfo()
            
            return {
                "MODEL NAME": self.model.getLp().model_name_,
                "STATUS": status_str,
                "VALOR ÓTIMO PRIMAL": f"{primal_value:e}" if primal_value is not None else "N/A",
                "VALOR ÓTIMO DUAL": f"{dual_value:e}" if dual_value is not None else "N/A",
                "GAP": f"{gap:e}" if gap is not None else "N/A",
                "INVIABILIDADE PRIMAL": f"{info.sum_primal_infeasibilities:e}",
                "INVIABILIDADE DUAL": f"{info.sum_dual_infeasibilities:e}",
                "ITERAÇÕES": info.simplex_iteration_count,
                "TEMPO(SEG.)": pc() - self.start_time
            }
        
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            return None