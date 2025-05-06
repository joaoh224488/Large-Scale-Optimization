import sys
import logging
import highspy as hs
import numpy as np

from time import perf_counter as pc
from highspy import Highs

import pandas as pd

class HighsSolver:
    # Inicializa o solver HiGHS e define o caminho do arquivo MPS
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.model = Highs()
        self.log_file = f'{instance_path.split("/")[-1].split(".")[0]}.log'
        self.model.setOptionValue("solver", "ipm")
        self.model.setOptionValue("log_file", self.log_file)
        self.res = None

    # Recebe o caminho para o arquivo MPS, lê o arquivo, executa o solver e armazena o resultado
    def run(self):
        try:
            # Carregar o modelo a partir do arquivo MPS
            status = self.model.readModel(self.instance_path)
            
            if status != hs.HighsStatus.kOk:
                raise Exception("Erro ao carregar o modelo MPS.")
            
            # Resolver o problema de otimização
            self.start_time = pc()
            self.model.run()
            self.res = self.model.getSolution()
            self.log_file = open(self.log_file, 'r', encoding='utf-8').read()
        
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
            dual = self.log_file.split('*')[-1].split()[3]
            base, ex = dual.split('e')
            dual_value = float(base)*10**int(ex)
            abs_gap = primal_value - dual_value
            rel_gap = self.log_file.split('Relative P-D gap')[-1].strip().split(':')[1].split('\n')[0].strip()
            primal_infeasibility = self.log_file.split('primal infeasibility:')[-1].split('\n')[0].strip()
            dual_infeasibility = self.log_file.split('dual infeasibility:')[-1].split('\n')[0].strip()
            
            if status == hs.HighsModelStatus.kOptimal:
                solution = self.model.getSolution()
                #dual_vars = solution.row_dual
                
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
                
                var_names = lp_model.col_names_
                restr_names = lp_model.row_names_  

                table1_data = {'VAR.': var_names,
                            'SOL. PRIMAL': solution.col_value,
                            'DUAL_PRICES': solution.col_dual # Custos reduzidos (associados às variáveis)
                            }
                df1 = pd.DataFrame(table1_data)  
            
                table2_data = {'RESTR.': restr_names,
                        'FOLGAS': solution.row_value,
                        'SOL. DUAL': solution.row_dual # Custos reduzidos (associados às variáveis)
                        }
                df2 = pd.DataFrame(table2_data)
                
                # Métricas de inviabilidade
                info = self.model.getInfo()
                
            return {
                "MODEL NAME": self.model.getLp().model_name_,
                "STATUS": status_str,
                "VALOR ÓTIMO PRIMAL": f"{primal_value:e}" if primal_value is not None else "N/A",
                "VALOR ÓTIMO DUAL": f"{dual_value:e}" if dual_value is not None else "N/A",
                "GAP ABSOLUTO": f"{abs_gap:e}" if abs_gap is not None else "N/A",
                "GAP RELATIVO": f"{rel_gap}" if rel_gap is not None else "N/A",
                "INVIABILIDADE PRIMAL": f"{primal_infeasibility}",
                "INVIABILIDADE DUAL": f"{dual_infeasibility}",
                "ITERAÇÕES": info.ipm_iteration_count,
                "TEMPO(SEG.)": pc() - self.start_time,
                "Df1": df1.set_index('VAR.'),
                "Df2": df2.set_index('RESTR.')
                }
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            return None