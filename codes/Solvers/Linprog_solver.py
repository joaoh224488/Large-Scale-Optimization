import sys
import logging
import highspy

from highspy import Highs
from scipy.optimize import linprog
from codes.read_instance_regex import MPSParser

class LinprogSolver:
    """
    Classe para resolver problemas de programação linear usando o método linprog do SciPy.

    Esta classe encapsula a funcionalidade de resolver problemas de programação linear
    que estão no formato MPS usando o solver linprog da biblioteca SciPy.

    Atributos:
        instance_path (str): Caminho para o arquivo MPS de entrada
        parser (MPSParser): Parser para ler o arquivo MPS
        data (dict): Dados do problema extraídos do arquivo MPS
        res (OptimizeResult): Resultado da otimização retornado pelo linprog

    Métodos:
        run(): Executa o solver linprog com os dados do problema
        print_results(): Imprime os resultados da otimização
        get_results(): Retorna um dicionário com os resultados da otimização
    """

    def __init__(self, instance_path):
        """
        Inicializa o solver com o caminho do arquivo MPS.

        Args:
            instance_path (str): Caminho para o arquivo MPS a ser resolvido
        """
        self.instance_path = instance_path
        self.parser = MPSParser(instance_path)
        self.data = self.parser.parse()
        self.res = None

    def run(self):
        """
        Executa o solver linprog com os dados do problema.

        Extrai os dados necessários do dicionário self.data e chama o método
        linprog do SciPy para resolver o problema de otimização.
        """
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
        """
        Imprime os resultados da otimização.

        Mostra o status da otimização, valor objetivo, sucesso e número de iterações.
        Se não houver resultados disponíveis ou ocorrer um erro, trata apropriadamente.
        """
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
        """
        Retorna um dicionário com os resultados da otimização.

        Returns:
            dict: Dicionário contendo status, valor objetivo, sucesso e número de iterações
                 ou None se não houver resultados ou ocorrer erro
        """
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