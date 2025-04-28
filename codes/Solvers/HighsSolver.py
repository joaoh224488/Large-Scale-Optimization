import sys
import logging
import highspy

from highspy import Highs

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
