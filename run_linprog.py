from read_instance_regex import MPSParser
from scipy.optimize import linprog
import numpy as np
import logging

class linprog_solver:
    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.parser = MPSParser(instance_path)
        self.data = self.parser.parse()
        self.res = None

    def run(self):
        print(f"Keys do objeto data: {self.data.keys()}")
        
        # Converter o dicionário A em uma matriz numpy
        rows = self.data['rows']
        cols = sorted(self.data['A'].keys())  # Ordenar as colunas para consistência
        n_rows = len(rows)
        n_cols = len(cols)
        
        # Criar matriz A preenchida com zeros
        A = np.zeros((n_rows, n_cols))
        
        # Preencher a matriz A com os valores do dicionário
        for j, col in enumerate(cols):
            for row_name, value in self.data['A'][col].items():
                i = rows.index(row_name)
                A[i, j] = value
        
        # Converter rhs em vetor numpy
        rhs = np.zeros(n_rows)
        for i, row in enumerate(rows):
            if row in self.data['rhs']:
                rhs[i] = self.data['rhs'][row]
        
        # Extrair coeficientes da função objetivo
        try:
            obj_row_idx = rows.index('OBJECTIV')
            c = A[obj_row_idx, :]
            
            # Remover a linha OBJECTIV da matriz A e do vetor rhs
            mask = np.array([row != 'OBJECTIV' for row in rows])
            A = A[mask]
            rhs = rhs[mask]
            rows = [row for row in rows if row != 'OBJECTIV']
            
            # Separar restrições de igualdade e desigualdade
            eq_mask = np.array([row.startswith('E') for row in rows])
            A_eq = A[eq_mask]
            b_eq = rhs[eq_mask]
            A_ub = A[~eq_mask]
            b_ub = rhs[~eq_mask]
            
            self.res = linprog(
                c=c,
                A_eq=A_eq, b_eq=b_eq,
                A_ub=A_ub, b_ub=b_ub,
                bounds=self.data.get('bounds', None),
                method="highs"
            )
        except ValueError as e:
            print(f"Erro ao processar função objetivo: {e}")
            print("Verifique se existe a linha 'OBJECTIV' no arquivo MPS")

    def print_results(self):

        try:
            print(f"Status: {self.res.message}")
            print(f"Valor objetivo: {self.res.fun}")
            print(f"Sucesso: {self.res.success}")
            print(f"Número de iterações: {self.res.nit}")
        except:
           raise Exception("Erro ao imprimir resultados")




def main():
   
   ls = linprog_solver("/home/vaio/ufpb/Large-Scale-Optimization/Instancias/agg.mps")
   ls.run()
   ls.print_results()
   
    

if __name__ == "__main__":
    main()

