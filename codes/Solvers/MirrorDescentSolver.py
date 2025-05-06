import re
import numpy as np
import pandas as pd
from time import perf_counter as pc
from scipy.sparse import coo_matrix
from typing import Tuple


class MirrorDescentSolver:
    def __init__(self, instance_path: str, eta=1e-1, epsilon=1e-5, max_iter=100000, versao="norma_p"):
        self.instance_path = instance_path
        self.eta = eta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.versao = versao

        self.A = None
        self.b = None
        self.x_star = None
        self.solution_cost = None
        self.solution = None
        self.time_taken = None

    def _read_instance(self) -> Tuple[coo_matrix, np.ndarray, np.ndarray, float]:
        with open(self.instance_path, 'r') as arquivo:
            texto = arquivo.read()

        def extrair_secao(padrao: str, texto: str) -> str:
            match = re.search(padrao, texto, re.DOTALL)
            if not match:
                raise ValueError(f"Seção não encontrada: {padrao}")
            return match.group(1).strip()

        padroes = {
            'row': r'matriz\.row : \n(.*?)(?=\nmatriz\.col :)',
            'col': r'matriz\.col : \n(.*?)(?=\nmatriz\.Data :)',
            'data': r'matriz\.Data : \n(.*?)(?=Vector b :)',
            'b': r'Vector b : (.*?)(?=X_star :)',
            'x_star': r'X_star : \n(.*?)(?=Objective Function Value :)',
            'solution_cost': r'Objective Function Value : (.*)'
        }

        secoes = {chave: extrair_secao(padroes[chave], texto) for chave in padroes}

        def parse_dados(texto: str, dtype) -> np.ndarray:
            numeros = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', texto)
            return np.array(list(map(dtype, numeros)))

        row = parse_dados(secoes['row'], int)
        col = parse_dados(secoes['col'], int)
        data = parse_dados(secoes['data'], float)
        b = parse_dados(secoes['b'], float)
        x_star = parse_dados(secoes['x_star'], float)
        solution_cost = float(secoes['solution_cost'])

        matriz = coo_matrix((data, (row, col)), shape=(len(b), len(b)), dtype=float)
        return matriz, b, x_star, solution_cost

    def _negativa_entropia(self, x):
        n = len(x)
        sum_ = 0

        for i in range(n):
            sum_ += x[i] * np.log(x[i]) - x[i]

        return  sum_

    def _grad_negativa_entropia(self, x):
        return np.log(x)

    def _inv_grad_negativa_entropia(self, x):
        return np.exp(x)

    def _norma_p(self, x, p = 2): # norma ao quadrado sobre 2
        return (1/p) * np.linalg.norm(x) ** p

    def _grad_norma_p(self, x, p = 2): # o gradiente dá o próprio x
        return np.sign(x) * (np.abs(x) ** (p - 1))

    def _inv_grad_norma_p(self, x, p = 2): # a inversa é o próprio x
        return np.sign(x) * (np.abs(x) ** (1 / (p-1)))

    def _grad_f(self, x):
        return self.A.T @ (self.A @ x -self.b)

    def f(self, x:np.array):
        return (1/2) * np.linalg.norm(self.A @ x - self.b)**2

    def mirror_gradient(self, x0, verbose=False, versao="norma_p"):
        xt = x0.copy()

        for i in range(self.max_iter):
            grad_f = self._grad_f(xt)
        
            if np.linalg.norm(grad_f) <= self.epsilon:
                break

            if versao == "negativa_entropia":
                x_new = self._inv_grad_negativa_entropia(
                    self._grad_negativa_entropia(xt) - self.eta * grad_f
                )
            elif versao == "norma_p":
                x_new = self._inv_grad_norma_p(
                    self._grad_norma_p(xt) - self.eta * grad_f
                )

            if np.linalg.norm(x_new - xt) < self.epsilon:
                break

            xt = x_new
            if verbose:
                print(np.linalg.norm(self._grad_f(xt)))
        return xt

    def run(self):
        try:
            self.A, self.b, self.x_star, self.solution_cost = self._read_instance()
            
            x0 = np.ones_like(self.b)

            start = pc()
            self.solution = self.mirror_gradient(x0, versao=self.versao)
            self.time_taken = pc() - start
        except Exception as e:
            print(f"Erro durante a execução: {e}")
            self.solution = None

    def get_results(self):
        if self.solution is None:
            return None
        try:
            grad_norm = np.linalg.norm(self._grad_f(self.solution))
            obj_val = self.f(self.solution)
            erro = 100 * (obj_val - self.solution_cost) / (self.solution_cost) 
            df = pd.DataFrame({
                'VAR': [f'x{i}' for i in range(len(self.solution))],
                'VALOR': self.solution,
                'x*': self.x_star,
                'ERRO': np.abs(self.solution - self.x_star)
            }).set_index('VAR')

            return { 
                "NORMA DO GRADIENTE": grad_norm,
                "CUSTO OBJETIVO CALCULADO": obj_val,
                "CUSTO OBJETIVO ESPERADO": self.solution_cost,
                "GAP COM RELAÇÃO AO OTIMO": erro,
                "TEMPO (s)": self.time_taken,
                "Df": df
            }
        except Exception as e:
            print(f"Erro ao processar os resultados: {e}")
            return None

    def print_results(self):
        results = self.get_results()
        if results is None:
            print("Nenhum resultado disponível.")
            return

        for k, v in results.items():
            if isinstance(v, pd.DataFrame):
                print(f"\n{k}:\n{v}")
            else:
                print(f"{k}: {v}")
