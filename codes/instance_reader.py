import re
import numpy as np
from scipy.sparse import coo_matrix
from typing import Tuple
import os

def read_instance(filename: str) -> Tuple[coo_matrix, np.ndarray, np.ndarray, float]:
    with open(filename, 'r') as arquivo:
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


def main():

    relative_path = "/toy/lsq_10000_1e-04_1.txt"

    # Remove leading slash to treat as relative path

    if relative_path.startswith("/"):

        relative_path = relative_path[1:]

    # Construct full path relative to current directory

    full_path = os.path.join(os.getcwd(), relative_path)

    

    if os.path.exists(full_path):

        print(f"The file exists at: {full_path}")

        try:

            matriz, b, x_star, solution_cost = read_instance(full_path)

            print("Matrix:", matriz)

            print("Vector b:", b)

            print("X_star:", x_star)

            print("Solution Cost:", solution_cost)

        except Exception as e:

            print(f"An error occurred while reading the instance: {e}")

    else:

        print(f"The file does not exist at: {full_path}")


if __name__ == "__main__":

    main()