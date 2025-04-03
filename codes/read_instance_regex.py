import re
import numpy as np
import logging
import sys

class MPSParser:

    """

    

    Classe para fazer parsing de arquivos no formato MPS (Mathematical Programming System).
    O formato MPS é um formato padrão da indústria para representar problemas de programação linear.

    Atributos:
        file_path (str): Caminho para o arquivo MPS a ser processado
        name (str): Nome do problema extraído do arquivo
        rows (list): Lista de tuplas (tipo, nome) representando as restrições
        objective_row (str): Nome da linha que representa a função objetivo
        A (dict): Dicionário para armazenar os coeficientes da matriz de restrições
        rhs (dict): Dicionário para armazenar os valores do lado direito das restrições
        bounds (dict): Dicionário para armazenar os limites das variáveis

    Métodos:
        extract_name(): Extrai o nome do problema do arquivo MPS
        extract_rows(): Extrai as informações da seção ROWS
        extract_columns(): Extrai os coeficientes da matriz de restrições
        extract_rhs(): Extrai os valores do lado direito das restrições
        extract_bounds(): Extrai os limites das variáveis
        parse(): Executa todo o processo de parsing do arquivo

    
    
    """



    def __init__(self, file_path):

        """
        Inicializa um novo parser MPS.

        Args:
            file_path (str): Caminho para o arquivo MPS a ser processado.

        Atributos inicializados:
            file_path (str): Caminho do arquivo
            name (str): Nome do problema (inicialmente vazio)
            rows (list): Lista de restrições (inicialmente vazia) 
            objective_row (str): Nome da função objetivo (inicialmente None)
            A (dict): Matriz de coeficientes (inicialmente vazio)
            rhs (dict): Valores do lado direito (inicialmente vazio)
            bounds (dict): Limites das variáveis (inicialmente vazio)
        """


        self.file_path = file_path
        self.name = ""
        self.rows = []
        self.objective_row = None
        self.A = {}
        self.rhs = {}
        self.bounds = {}
    
    def extract_name(self):
        
        """
        Extrai o nome do problema do arquivo MPS.

        O nome é encontrado na primeira linha do arquivo que começa com "NAME".
        O formato esperado é "NAME [nome_do_problema]".

        Returns:
            str: Nome do problema extraído do arquivo
        """

        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith("NAME"):
                    self.name = line.split()[1]
                    break
        return self.name
    
    def extract_rows(self):

        """
        Extrai as informações da seção ROWS do arquivo MPS.

        A seção ROWS contém os tipos e nomes das restrições.
        Cada linha tem o formato "[tipo] [nome]" onde:
        - tipo pode ser:
            N: função objetivo
            E: restrição de igualdade (=)
            L: restrição menor ou igual (<=) 
            G: restrição maior ou igual (>=)
        - nome é o identificador da restrição

        Returns:
            list: Lista de tuplas (tipo, nome) para cada restrição
        """


        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("ROWS\n") + 1
            end = lines.index("COLUMNS\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) == 2:
                    row_type, row_name = parts
                    if row_type == "N":
                        self.objective_row = row_name  # Identifica a função objetivo
                    self.rows.append((row_type, row_name))
        return self.rows
    
    def extract_columns(self):

        """
        Extrai as informações da seção COLUMNS do arquivo MPS.

        A seção COLUMNS contém os coeficientes da matriz A do problema.
        Cada linha tem o formato "[nome_coluna] [nome_linha] [valor]" ou
        "[nome_coluna] [nome_linha1] [valor1] [nome_linha2] [valor2]" onde:
        - nome_coluna: nome da variável
        - nome_linha: nome da restrição 
        - valor: coeficiente da variável na restrição

        Returns:
            dict: Dicionário com os coeficientes da matriz A, onde:
                 - chave externa é o nome da coluna (variável)
                 - chave interna é o nome da linha (restrição)
                 - valor é o coeficiente
        """

        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("COLUMNS\n") + 1
            end = lines.index("RHS\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) >= 3:
                    col_name, row_name, value = parts[:3]
                    value = float(value)
                    if col_name not in self.A:
                        self.A[col_name] = {}
                    self.A[col_name][row_name] = value
                    
                    if len(parts) == 5:
                        row_name2, value2 = parts[3:]
                        value2 = float(value2)
                        self.A[col_name][row_name2] = value2
        return self.A
    
    def extract_rhs_in(self):



        
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("RHS\n") + 1
            end = lines.index("ENDATA\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) >= 3:
                    # Extrai o nome da linha e o valor, ignorando o primeiro elemento
                    _, row_name, value = parts[:3]
                    # Converte o valor para float
                    value = float(value)
                    # Armazena o valor no dicionário rhs usando o nome da linha como chave
                    self.rhs[row_name] = value
                    
                    if len(parts) == 5:
                        row_name2, value2 = parts[3:]
                        value2 = float(value2)
                        self.rhs[row_name2] = value2
        return self.rhs
    
    def extract_rhs(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("RHS\n") + 1
            end = lines.index("ENDATA\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        _, row_name, value = parts[:3]
                        value = float(value)  # Tentar converter para float
                        self.rhs[row_name] = value
                        
                        if len(parts) == 5:
                            row_name2, value2 = parts[3:]
                            value2 = float(value2)  # Tentar converter para float
                            self.rhs[row_name2] = value2
                    except ValueError:
                        # Ignorar valores que não podem ser convertidos para float
                        logging.warning(f"Valor inválido na linha RHS: {line.strip()}")
                        continue
        return self.rhs


    def extract_bounds(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            if "BOUNDS\n" in lines:
                start = lines.index("BOUNDS\n") + 1
                end = lines.index("ENDATA\n")
                
                for line in lines[start:end]:
                    parts = line.split()
                    if len(parts) >= 3:
                        bound_type, col_name, value = parts[:3]
                        value = float(value)
                        if col_name not in self.bounds:
                            self.bounds[col_name] = {}
                        self.bounds[col_name][bound_type] = value
        return self.bounds
    
    def parse(self):
        self.extract_name()
        self.extract_rows()
        self.extract_columns()
        self.extract_rhs()
        self.extract_bounds()
        
        # Extrair coeficientes da função objetivo
        c = []
        variables = sorted(self.A.keys())
        for var in variables:
            if self.objective_row in self.A[var]:
                c.append(self.A[var][self.objective_row])
            else:
                c.append(0.0)  # Se a variável não aparece na função objetivo, coeficiente é 0
        
        # Organizar restrições
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        for row_type, row_name in self.rows:
            if row_name == self.objective_row:
                continue  # Ignorar a função objetivo
            
            row_coeffs = []
            for var in variables:
                if row_name in self.A[var]:
                    row_coeffs.append(self.A[var][row_name])
                else:
                    row_coeffs.append(0.0)
            
            if row_type == "L":  # Restrição de desigualdade <=
                A_ub.append(row_coeffs)
                b_ub.append(self.rhs.get(row_name, 0.0))
            elif row_type == "E":  # Restrição de igualdade
                A_eq.append(row_coeffs)
                b_eq.append(self.rhs.get(row_name, 0.0))
            elif row_type == "G":  # Restrição de desigualdade >=
                A_ub.append([-x for x in row_coeffs])  # Transformar em <=
                b_ub.append(-self.rhs.get(row_name, 0.0))
        
        # Converter para arrays numpy
        c = np.array(c)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Extrair limites das variáveis
        bounds = []
        for var in variables:
            var_bounds = self.bounds.get(var, {})
            lower = var_bounds.get("LO", 0.0)  # Valor padrão para limite inferior
            upper = var_bounds.get("UP", np.inf)  # Valor padrão para limite superior
            bounds.append((lower, upper))
        
        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "variables": variables
        }

def main():
    # Verifica se um arquivo foi passado como argumento
    if len(sys.argv) < 2:
        print("Exemplo de uso: python *seu_codigo*.py *seu arquivo*.mps")
        sys.exit(1)  # Encerra o programa se nenhum arquivo for fornecido

    # Obtém o caminho do arquivo do primeiro argumento
    mps_file = sys.argv[1]

    try:
        # Cria o parser e processa o arquivo
        parser = MPSParser(mps_file)
        data = parser.parse()

        # Exibe os resultados
        print("Coeficientes da função objetivo (c):", data["c"])
        print("Matriz de restrições de desigualdade (A_ub):", data["A_ub"])
        print("Vetor do lado direito das restrições de desigualdade (b_ub):", data["b_ub"])
        print("Matriz de restrições de igualdade (A_eq):", data["A_eq"])
        print("Vetor do lado direito das restrições de igualdade (b_eq):", data["b_eq"])
        print("Limites das variáveis (bounds):", data["bounds"])
        print("Variáveis:", data["variables"])
    except FileNotFoundError:
        print(f"Erro: Arquivo '{mps_file}' não encontrado.")
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    main()