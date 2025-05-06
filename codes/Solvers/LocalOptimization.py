import sys
import logging
import numpy as np
from scipy.optimize import minimize,least_squares,OptimizeResult
from scipy.sparse import coo_matrix
from typing import Tuple

class LocalOptimization:
    """
    Classe para resolver problemas de otimização local não-linear usando métodos do SciPy.
    Especializada para problemas de mínimos quadrados: 1/2 * ||A*x - b||².

    Atributos:
        A (coo_matrix): Matriz esparsa do problema
        b (np.ndarray): Vetor do lado direito
        bounds (list): Limites das variáveis [(min1, max1), (min2, max2), ...]
        constraints (dict): Restrições não-lineares (para métodos que suportam)
        default_method (str): Método padrão ('nelder_mead', 'powell', 'bfgs', etc.)
        result (OptimizeResult): Resultado da otimização
    """

    def __init__(self, A: coo_matrix, b: np.ndarray, bounds=None, constraints=None, default_method=''):
        """
        Inicializa o otimizador local para problemas de mínimos quadrados.

        Args:
            A (coo_matrix): Matriz esparsa do problema
            b (np.ndarray): Vetor do lado direito
            bounds (list, optional): Lista de tuplas com limites para cada variável
            constraints (dict, optional): Restrições no formato {'type': 'ineq/eq', 'fun': callable}
            default_method (str): Método padrão ('nelder_mead', 'powell', etc.)
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
        return (np.linalg.norm(self.A*x-self.b)**2)*0.5

    def run(self, method=None):
        """
        Executa o método de otimização especificado.

        Args:
            method (str, optional): Nome do método ('nelder_mead', 'powell', etc.)
                                    Se None, usa o método padrão.
        Returns:
            OptimizeResult: Resultado da otimização
        """
        method = method if method is not None else self.default_method
        method = method.lower()

        try:
            if method == 'nelder_mead':
                self.result = self._run_nelder_mead()
            elif method == 'lsq':
                self.result = self._run_least_squares()
            elif method == 'powell':
                self.result = self._run_powell()
            elif method == 'cg':
                self.result = self._run_cg()
            elif method == 'bfgs':
                self.result = self._run_bfgs()
            elif method == 'newton_cg':
                self.result = self._run_newton_cg()
            elif method == 'l_bfgs_b':
                self.result = self._run_l_bfgs_b()
            elif method == 'tnc':
                self.result = self._run_tnc()
            elif method == 'cobyla':
                self.result = self._run_cobyla()
            elif method == 'slsqp':
                self.result = self._run_slsqp()
            elif method == 'trust_constr':
                self.result = self._run_trust_constr()
            elif method == 'dogleg':
                self.result = self._run_dogleg()
            elif method == 'trust_ncg':
                self.result = self._run_trust_ncg()
            elif method == 'trust_exact':
                self.result = self._run_trust_exact()
            elif method == 'trust_krylov':
                self.result = self._run_trust_krylov()
            else:
                raise ValueError(f"Método '{method}' não suportado. Use: 'nelder_mead', 'powell', 'bfgs', 'l_bfgs_b', etc.")
            
            return self.result
        
        except Exception as e:
            logging.error(f"Erro na execução do método {method}: {e}")
            self.result = None
            return None

    def _run_nelder_mead(self):
        """Executa o algoritmo Nelder-Mead."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='Nelder-Mead'
        )

    def _run_powell(self):
        """Executa o algoritmo Powell."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='Powell'
        )

    def _run_cg(self):
        """Executa o algoritmo CG (conjugate gradient)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='CG'
        )

    def _run_bfgs(self):
        """Executa o algoritmo BFGS (Broyden-Fietcher-Goldfarb-Shanno)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='BFGS'
        )

    def _run_newton_cg(self):
        """Executa o algoritmo Newton-CG (Newton-conjugate gradient)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='Newton-CG'
        )

    def _run_l_bfgs_b(self):
        """Executa o algoritmo L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Box."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='L-BFGS-B',
            bounds=self.bounds
        )

    def _run_tnc(self):
        """Executa o algoritmo TNC (Truncated Newton conjugate Gradient)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='TNC',
            bounds=self.bounds
        )

    def _run_cobyla(self):
        """Executa o algoritmo COBYLA (Constrained Optimization by Linear Aproximations)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='COBYLA',
            constraints=self.constraints if self.constraints else None
        )

    def _run_slsqp(self):
        """Executa o algoritmo SLSQP (Sequential Least Squares Quadratic Programming)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints if self.constraints else None
        )

    def _run_trust_constr(self):
        """Executa o algoritmo trust-constr (Trust-constraints)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='trust-constr',
            bounds=self.bounds,
            constraints=self.constraints if self.constraints else None
        )

    def _run_dogleg(self):
        """Executa o algoritmo dogleg."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='dogleg'
        )

    def _run_trust_ncg(self):
        """Executa o algoritmo trust-ncg (Trust Nonlinear Conjugate Gradient)."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='trust-ncg'
        )

    def _run_trust_exact(self):
        """Executa o algoritmo trust-exact."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='trust-exact'
        )

    def _run_trust_krylov(self):
        """Executa o algoritmo trust-krylov."""
        return minimize(
            fun=self.objective_function,
            x0=np.zeros(self.A.shape[1]),
            method='trust-krylov'
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
            
            if hasattr(self.result, 'njev'):
                print(f"Avaliações do gradiente: {self.result.njev}")
            
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
            
            return results_dict
        
        except Exception as e:
            logging.error(f"Erro ao obter resultados: {e}")
            logging.error(f"RESULT OBJECT: {vars(self.result) if hasattr(self.result, '__dict__') else self.result}")
            return None

# Exemplo de uso
if __name__ == "__main__":
    import os
    from instance_reader import read_instance

    # Configuração do problema
    relative_path = "toy/lsq_1000_1e-04_1.txt"
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
            optimizer = LocalOptimization(A, b, bounds=bounds)
            
            # Executar todos os métodos e comparar com a solução conhecida
            methods = ['lsq','l_bfgs_b', 'bfgs'] 
            
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
