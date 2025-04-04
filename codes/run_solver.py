import sys
import logging
import highspy

from highspy import Highs
from scipy.optimize import linprog
from codes.read_instance_regex import MPSParser


class HighsSolver:
    """
    Classe para resolver problemas de programação linear usando o solver HiGHS.

    Esta classe encapsula a funcionalidade do solver HiGHS, permitindo carregar
    e resolver problemas de programação linear no formato MPS.

    Atributos:
        instance_path (str): Caminho para o arquivo MPS de entrada
        model (Highs): Instância do solver HiGHS
        res (HighsSolution): Resultado da otimização após resolver o problema

    Métodos:
        run(): Carrega e resolve o problema de otimização
        print_results(): Imprime os resultados da otimização no console
        get_results(): Retorna um dicionário com os resultados da otimização

    O HiGHS é um solver de código aberto para programação linear que oferece:
    - Alta performance
    - Estabilidade numérica
    - Capacidade de resolver problemas de grande escala
    - Interface Python via highspy
    """

    def __init__(self, instance_path):
        """
        Inicializa o solver HiGHS.

        Args:
            instance_path (str): Caminho para o arquivo MPS que será resolvido

        Atributos inicializados:
            instance_path: Armazena o caminho do arquivo
            model: Cria uma nova instância do solver HiGHS
            res: Armazena o resultado da otimização (inicialmente None)
        """
        self.instance_path = instance_path
        self.model = Highs()
        self.res = None

    def run(self):
        """
        Executa o solver HiGHS para resolver o problema.

        Este método:
        1. Carrega o modelo do arquivo MPS usando readModel()
        2. Verifica se o carregamento foi bem sucedido
        3. Executa o solver usando run()
        4. Obtém a solução usando getSolution()

        Em caso de erro:
        - Registra o erro no log
        - Define self.res como None
        """
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
            print(f"Status: {self.model.modelStatusToString(self.model.getModelStatus())}")
            print(f"Valor objetivo: {self.model.getObjectiveValue()}")
            print(f"success: {self.model.getModelStatus()}")
            print(f"Número de iterações: {self.model.getInfo().simplex_iteration_count}")
        
        except Exception as e:
            raise Exception(f"Erro ao imprimir resultados: {e}")
    
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