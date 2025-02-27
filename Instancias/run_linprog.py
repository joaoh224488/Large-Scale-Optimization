from read_instance_regex import MPSParser
from scipy.optimize import linprog
import numpy as np

def main():
   
   
    parser = MPSParser("/home/vaio/ufpb/Large-Scale-Optimization/Instancias/25fv47.mps")
    data = parser.parse()

    # Exibir as matrizes resultantes
    # print("Nome do Problema:", data["name"])
    # print("Vetor c (Objetivo):", data["c"])
    # print("Matriz A_eq (Igualdades):", data["A_eq"])
    # print("Vetor b_eq:", data["b_eq"])
    # print("Matriz A_ub (Inequações ≤):", data["A_ub"])
    # print("Vetor b_ub:", data["b_ub"])
    # print("Limites das Variáveis:", data["bounds"])
    print("OPtimize")
 

    res = linprog(
        c=data["c"],
        A_eq=data["A_eq"], b_eq=data["b_eq"],
        A_ub=data["A_ub"], b_ub=data["b_ub"],
        bounds=data["bounds"],
        method="highs"  # Melhor solver disponível no scipy
    )

    # print(res)

    
    # Imprime os resultados
    print("Status:", res.message)
    print("Valor objetivo:", res.fun)
    print("Sucesso:", res.success)
    print("Número de iterações:", res.nit)

if __name__ == "__main__":
    main()

