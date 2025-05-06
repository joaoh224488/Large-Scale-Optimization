import sys
import logging
import numpy as np
from scipy.optimize import basinhopping, differential_evolution, shgo, dual_annealing,minimize,least_squares,OptimizeResult
from scipy.sparse import coo_matrix
from typing import Tuple

class GlobalOptimization:
    """
    Classe para resolver problemas de otimização global não-linear usando métodos do SciPy.
    Especializada para problemas de mínimos quadrados: 1/2 * ||A*x - b||².

    Atributos:
        A (coo_matrix): Matriz esparsa do problema
        b (np.ndarray): Vetor do lado direito
        bounds (list): Limites das variáveis [(min1, max1), (min2, max2), ...]
        constraints (dict): Restrições não-lineares (para métodos que suportam)
        default_method (str): Método padrão ('basinhopping', 'diff_evolution', 'shgo', 'dual_annealing')
        result (OptimizeResult): Resultado da otimização
    """

    def __init__(self, A: coo_matrix, b: np.ndarray, bounds=None, constraints=None, default_method=''):
        """
        Inicializa o otimizador global para problemas de mínimos quadrados.

        Args:
            A (coo_matrix): Matriz esparsa do problema
            b (np.ndarray): Vetor do lado direito
            bounds (list, optional): Lista de tuplas com limites para cada variável
            constraints (dict, optional): Restrições no formato {'type': 'ineq/eq', 'fun': callable}
            default_method (str): Método padrão ('diff_evolution', 'basinhopping', etc.)
        """
        self.A = A
        self.b = b
        self.bounds = bounds if bounds is not None else []
        self.constraints = constraints if constraints is not None else {}
        self.default_method = default_method
        self.result = None



    def objective_function(self, x):

        """
        Função objetivo de mínimos quadrados: 1/2 * ||A*x - b||²
        """

        # residual = self.A.dot(x) - self.b  # Preserva a esparsidade
        # return 0.5 * residual.dot(residual)  # Mais eficiente que sum(residual**2)

        return (np.linalg.norm(self.A*x-self.b)**2)*0.5

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
            elif method == 'lsq':
                self.result = self._run_least_squares()
            else:
                raise ValueError(f"Método '{method}' não suportado. Use: 'basinhopping', 'diff_evolution', 'shgo', ou 'dual_annealing'")
            
            return self.result
        
        except Exception as e:
            logging.error(f"Erro na execução do método {method}: {e}")
            self.result = None
            return None
        


    def _run_basinhopping(self):
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds,
            'options': {
                'maxiter': 50,  # Reduzir iterações locais
                'ftol': 1e-4    # Tolerância mais relaxada
            }
        }
        
        return basinhopping(
            func=self.objective_function,
            x0=self.best_initial_guess(),  # Ver implementação abaixo
            minimizer_kwargs=minimizer_kwargs,
            niter=30,           # Reduzir iterações globais
            T=1.0,              # Temperatura inicial menor
            stepsize=0.5        # Passo menor
        )

    def best_initial_guess(self):
        """Retorna um chute inicial inteligente"""
        return np.zeros(self.A.shape[1])  # Ou outra heurística

    def _run_diff_evolution(self):
        return differential_evolution(
            func=self.objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            popsize=min(5, len(self.bounds)),  # População pequena
            maxiter=20,
            tol=0.01,
            polish=False,
            workers=-1,  # Paralelizar se possível
            updating='deferred'
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
    

    def _run_least_squares(self):
        """
        Resolve o problema de mínimos quadrados não-lineares usando scipy.optimize.least_squares.
        Ideal para problemas com n > 1000 variáveis e matriz A esparsa.
        
        Returns:
            OptimizeResult: Resultado da otimização no formato padronizado.
        """
        # Prepara os bounds no formato correto para least_squares
        if self.bounds:
            # Converte lista de tuplas para arrays separados de lower e upper bounds
            lb = np.array([b[0] for b in self.bounds])
            ub = np.array([b[1] for b in self.bounds])
            bounds = (lb, ub)
        else:
            bounds = (-np.inf, np.inf)  # Sem bounds definidos

        # Configurações do least_squares
        ls_kwargs = {
            'fun': lambda x: self.A.dot(x) - self.b,  # Função de resíduos A*x - b
            'x0': np.zeros(self.A.shape[1]),  # Vetor zero como chute inicial
            'bounds': bounds,
            'method': 'trf',  # Trust Region Reflective (para problemas esparsos)
            'ftol': 1e-6,     # Tolerância no valor da função
            'xtol': 1e-6,     # Tolerância nos parâmetros
            'gtol': 1e-6,     # Tolerância no gradiente
            'max_nfev': 1000,  # Máximo de avaliações da função
            'verbose': 0,      # 0-2 para nível de detalhe
            'tr_solver': 'lsmr',  # Usa LSMR para problemas esparsos grandes
            'loss': 'linear'   # Função de perda linear (mínimos quadrados padrão)
        }
        
        try:
            # Executa a otimização
            result = least_squares(**ls_kwargs)
            
            # Converte para o formato compatível com outros métodos da classe
            return OptimizeResult(
                x=result.x,
                success=result.success,
                status=result.status,
                message=result.message,
                fun=0.5 * np.sum(result.fun**2),  # Converte resíduos para 0.5*||r||^2
                nfev=result.nfev,
                njev=result.njev if hasattr(result, 'njev') else None
            )
        except Exception as e:
            logging.error(f"Erro no least_squares: {e}")
            raise


    def residual_function(self, x):
        """Retorna os resíduos A*x - b para least_squares"""
        return self.A.dot(x) - self.b

    def best_initial_guess(self):
        """Retorna um chute inicial inteligente (pode ser personalizado)"""
        return np.zeros(self.A.shape[1])


    def print_results(self):
        """Imprime os resultados resumidos da otimização."""
        if self.result is None:
            print("Nenhum resultado disponível. Execute o método run() primeiro.")
            return
        
        try:
            print("\n=== Resultados da Otimização ===")
            print(f"Status: {self.result.message}")
            print(f"Valor objetivo (f(x)): {self.result.fun}")
            print(f"Ponto ótimo (x): {self.result.x}")
            print(f"Sucesso: {self.result.success}")
            
            if hasattr(self.result, 'nit'):
                print(f"Iterações totais: {self.result.nit}")
            
            if hasattr(self.result, 'nfev'):
                print(f"Avaliações da função: {self.result.nfev}")
            
            if hasattr(self.result, 'minimization_failures'):
                print(f"Falhas em minimizações locais: {self.result.minimization_failures}")
            
            if hasattr(self.result, 'population'):
                print(f"Tamanho da população: {len(self.result.population)}")
        
        except Exception as e:
            logging.error(f"Erro ao imprimir resultados: {e}")
            logging.error(f"RESULT OBJECT: {vars(self.result) if hasattr(self.result, '__dict__') else self.result}")

    def get_results(self):
        """
        Retorna um dicionário com os resultados detalhados.
        """
        if self.result is None:
            return None
        
        try:
            results_dict = {
                'message': str(self.result.message),
                'fun': float(self.result.fun),
                'x': np.array(self.result.x).tolist(),
                'success': bool(self.result.success),
            }
            
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

# Exemplo de uso integrado com instance_reader
if __name__ == "__main__":
    import os
    from instance_reader import read_instance

    # Configuração do problema
    relative_path = "toy/lsq_1000_1e-04_1.txt"
    # relative_path = "instances_lsq_linear/lsq_10000_1e-03_1.txt"
    full_path = os.path.join(os.getcwd(), relative_path)

    if os.path.exists(full_path):
        print(f"Arquivo encontrado em: {full_path}")
        try:
            # Ler a instância
            A, b, x_star, solution_cost = read_instance(full_path)
            print(f"Dimensões: A={A.shape}, b={b.shape}")
            
            # Definir bounds (exemplo: -10 a 10 para todas as variáveis)
            n_vars = A.shape[1]
            bounds = [(-2, 2) for _ in range(n_vars)]
            
            # Criar otimizador
            optimizer = GlobalOptimization(A, b, bounds=bounds)
            
            # Executar todos os métodos e comparar com a solução conhecida
            # Modifique sua lista de métodos para excluir SHGO quando n_vars > 100
            # methods = ['hybrid','basinhopping', 'diff_evolution'] 
            methods = ['lsq']
            
            print(f"\nSolução conhecida (x_star) tem custo: {solution_cost}")
            
            for method in methods:
                print(f"\n=== Executando {method} ===")
                result = optimizer.run(method)
                if result:
                    optimizer.print_results()
                    print(f"Diferença para solução ótima: {abs(result.fun - solution_cost)}")
        except Exception as e:
            print(f"Erro ao processar a instância: {e}")
    else:
        print(f"Arquivo não encontrado: {full_path}")
