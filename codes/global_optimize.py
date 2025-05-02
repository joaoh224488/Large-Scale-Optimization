import sys
import logging
import numpy as np
from scipy.optimize import basinhopping, differential_evolution, shgo, dual_annealing

class GlobalOptimization:
    """
    Classe para resolver problemas de otimização global não-linear usando métodos do SciPy.

    Atributos:
        objective_function (callable): Função objetivo a ser minimizada
        bounds (list): Limites das variáveis [(min1, max1), (min2, max2), ...]
        constraints (dict): Restrições não-lineares (para métodos que suportam)
        default_method (str): Método padrão ('basinhopping', 'diff_evolution', 'shgo', 'dual_annealing')
        result (OptimizeResult): Resultado da otimização

    Métodos:
        run(method=None): Executa o método de otimização especificado
        print_results(): Imprime os resultados resumidos
        get_results(): Retorna um dicionário com os resultados detalhados
        _run_basinhopping(): Implementa basinhopping
        _run_diff_evolution(): Implementa differential_evolution
        _run_shgo(): Implementa shgo
        _run_dual_annealing(): Implementa dual_annealing
    """

    def __init__(self, objective_function, bounds=None, constraints=None, default_method='diff_evolution'):
        """
        Inicializa o otimizador global.

        Args:
            objective_function (callable): Função objetivo f(x) a ser minimizada
            bounds (list, optional): Lista de tuplas com limites para cada variável
            constraints (dict, optional): Restrições no formato {'type': 'ineq/eq', 'fun': callable}
            default_method (str): Método padrão ('diff_evolution', 'basinhopping', etc.)
        """
        self.objective_function = objective_function
        self.bounds = bounds if bounds is not None else []
        self.constraints = constraints if constraints is not None else {}
        self.default_method = default_method
        self.result = None

    def run(self, method=None):
        """
        Executa o método de otimização especificado.

        Args:
            method (str, optional): Nome do método ('basinhopping', 'diff_evolution', etc.)
                                    Se None, usa o método padrão.
        Returns:
            OptimizeResult: Resultado da otimização
        """
        method = method if method is not None else self.default_method
        method = method.lower()

        try:
            if method == 'basinhopping':
                self.result = self._run_basinhopping()
            elif method == 'diff_evolution':
                self.result = self._run_diff_evolution()
            elif method == 'shgo':
                self.result = self._run_shgo()
            elif method == 'dual_annealing':
                self.result = self._run_dual_annealing()
            else:
                raise ValueError(f"Método '{method}' não suportado. Use: 'basinhopping', 'diff_evolution', 'shgo', ou 'dual_annealing'")
            
            return self.result
        
        except Exception as e:
            logging.error(f"Erro na execução do método {method}: {e}")
            self.result = None
            return None

    def _run_basinhopping(self):
        """Executa o algoritmo Basin Hopping."""
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds,
        }
        
        if self.constraints:
            minimizer_kwargs['constraints'] = self.constraints

        return basinhopping(
            func=self.objective_function,
            x0=np.mean(self.bounds, axis=1) if self.bounds else [0],
            minimizer_kwargs=minimizer_kwargs,
            niter=100
        )

    def _run_diff_evolution(self):
        """Executa Differential Evolution."""
        if not self.bounds:
            raise ValueError("Differential evolution requer bounds para todas as variáveis.")
        
        return differential_evolution(
            func=self.objective_function,
            bounds=self.bounds,
            polish=True  # Refina o resultado com um método local
        )

    def _run_shgo(self):
        """Executa o algoritmo SHGO (Simplicial Homology Global Optimization)."""
        return shgo(
            func=self.objective_function,
            bounds=self.bounds,
            constraints=self.constraints if self.constraints else None,
            sampling_method='sobol'
        )

    def _run_dual_annealing(self):
        """Executa Dual Annealing."""
        if not self.bounds:
            raise ValueError("Dual annealing requer bounds para todas as variáveis.")
        
        return dual_annealing(
            func=self.objective_function,
            bounds=self.bounds,
            maxiter=1000
        )

    def print_results(self):
        """Imprime os resultados resumidos da otimização."""
        if self.result is None:
            print("Nenhum resultado disponível. Execute o método run() primeiro.")
            return
        
        try:
            # Métodos globais do SciPy não possuem o atributo 'method' diretamente
            # Extraímos informações com base na estrutura conhecida dos resultados
            print("\n=== Resultados da Otimização ===")
            
            # Mensagem de status (disponível em todos)
            print(f"Status: {self.result.message}")
            
            # Valor objetivo
            print(f"Valor objetivo (f(x)): {self.result.fun}")
            
            # Ponto ótimo
            print(f"Ponto ótimo (x): {self.result.x}")
            
            # Sucesso (booleano)
            print(f"Sucesso: {self.result.success}")
            
            # Informações específicas por método
            if hasattr(self.result, 'nit'):
                print(f"Iterações totais: {self.result.nit}")
            
            if hasattr(self.result, 'nfev'):
                print(f"Avaliações da função: {self.result.nfev}")
            
            # Para basinhopping, mostramos falhas de minimização local
            if hasattr(self.result, 'minimization_failures'):
                print(f"Falhas em minimizações locais: {self.result.minimization_failures}")
            
            # Para differential_evolution, mostramos tamanho da população
            if hasattr(self.result, 'population'):
                print(f"Tamanho da população: {len(self.result.population)}")
        
        except Exception as e:
            logging.error(f"Erro ao imprimir resultados: {e}")
            logging.error(f"RESULT OBJECT: {vars(self.result) if hasattr(self.result, '__dict__') else self.result}")

    def get_results(self):
        """
        Retorna um dicionário com os resultados detalhados.
        Adaptado para funcionar com todos os métodos globais do SciPy.
        """
        if self.result is None:
            return None
        
        try:
            results_dict = {
                'message': str(self.result.message),
                'fun': float(self.result.fun),
                'x': np.array(self.result.x).tolist(),  # Converte para lista se for array
                'success': bool(self.result.success),
            }
            
            # Adiciona atributos condicionais
            if hasattr(self.result, 'nit'):
                results_dict['nit'] = int(self.result.nit)
            
            if hasattr(self.result, 'nfev'):
                results_dict['nfev'] = int(self.result.nfev)
            
            if hasattr(self.result, 'njev'):
                results_dict['njev'] = int(self.result.njev)
            
            if hasattr(self.result, 'minimization_failures'):
                results_dict['minimization_failures'] = int(self.result.minimization_failures)
            
            return results_dict
        
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            logging.error(f"RESULT OBJECT: {vars(self.result) if hasattr(self.result, '__dict__') else self.result}")
            return None

# Exemplo de uso
if __name__ == "__main__":
    # Função objetivo de exemplo (Rastrigin)
    def rastrigin(x):
        return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

    # Configuração do problema
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # Bounds para 2 variáveis
    optimizer = GlobalOptimization(rastrigin, bounds=bounds)

    # Executa todos os métodos e imprime resultados
    methods = ['basinhopping', 'diff_evolution', 'shgo', 'dual_annealing']
    
    for method in methods:
        print(f"\n=== Executando {method} ===")
        optimizer.run(method)
        optimizer.print_results()
