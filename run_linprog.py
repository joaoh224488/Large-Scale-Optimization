from read_instance_regex import MPSParser
from scipy.optimize import linprog
import numpy as np
import logging
import sys

class linprog_solver:
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.parser = MPSParser(instance_path)
        self.data = self.parser.parse()
        self.res = None

    def run(self):
        #print(f"Keys do objeto data: {self.data.keys()}")
        
        # Extrair os dados diretamente do parser
        c = self.data["c"]
        A_ub = self.data["A_ub"]
        b_ub = self.data["b_ub"]
        A_eq = self.data["A_eq"]
        b_eq = self.data["b_eq"]
        bounds = self.data["bounds"]
        
        # Resolver o problema de programação linear
        self.res = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

    def print_results(self):
        try:
            print(f"Status: {self.res.message}")
            print(f"Valor objetivo: {self.res.fun}")
            print(f"Sucesso: {self.res.success}")
            print(f"Número de iterações: {self.res.nit}")

        except Exception as e:
            raise Exception(f"Erro ao imprimir resultados: {e}")
        
    def get_results(self):
        try:
            return {
                "status": self.res.message,
                "objective_value": self.res.fun,
                "success": self.res.success,
                "iterations": self.res.nit,
            }
        
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            return None

def main():
    if len(sys.argv) < 2:
        print("Exemplo de uso: python *seu_codigo*.py *seu arquivo*.mps")
        sys.exit(1)
    
    mps_file = sys.argv[1]

    try:
        ls = linprog_solver(mps_file)
        ls.run()
        ls.print_results()

    except FileNotFoundError:
        print(f"Erro: Arquivo '{mps_file}' não encontrado.")

    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()