import sys
import logging
import highspy

from highspy import Highs
from scipy.optimize import linprog
from codes.read_instance_regex import MPSParser

class HighsSolver:
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.model = Highs()
        self.res = None

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


class LinprogSolver:
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.parser = MPSParser(instance_path)
        self.data = self.parser.parse()
        self.res = None

    def run(self):
        c = self.data["c"]
        A_ub = self.data["A_ub"]
        b_ub = self.data["b_ub"]
        A_eq = self.data["A_eq"]
        b_eq = self.data["b_eq"]
        bounds = self.data["bounds"]
        
        self.res = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

    def print_results(self):
        if self.res is None:
            print("Nenhum resultado disponível.")
            return
        
        try:
            print(f"Status: {self.res.message}")
            print(f"Valor objetivo: {self.res.fun}")
            print(f"Sucesso: {self.res.success}")
            print(f"Número de iterações: {self.res.nit}")
        
        except Exception as e:
            raise Exception(f"Erro ao imprimir resultados: {e}")
    
    def get_results(self):
        if self.res is None:
            return None
        
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
    if len(sys.argv) < 3:
        print("Uso: python run_solver.py -h|-l arquivo.mps")
        sys.exit(1)
    
    solver_flag = sys.argv[1]
    mps_file = sys.argv[2]
    
    if solver_flag == "-h":
        solver = HighsSolver(mps_file)
    
    elif solver_flag == "-l":
        solver = LinprogSolver(mps_file)
    
    else:
        print("Opção inválida. Use -h para Highs ou -l para linprog.")
        sys.exit(1)
    
    try:
        solver.run()
        solver.print_results()
    
    except FileNotFoundError:
        print(f"Erro: Arquivo '{mps_file}' não encontrado.")
    
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()