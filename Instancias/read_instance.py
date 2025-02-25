import numpy as np

class MPSReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.c = None
        self.A_ub = []
        self.b_ub = []
        self.A_eq = []
        self.b_eq = []
        self.bounds = []
        self.var_names = []

    def read_mps(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        section = None
        for line in lines:
            line = line.strip()
            if line.startswith('NAME'):
                continue
            elif line.startswith('ROWS'):
                section = 'ROWS'
            elif line.startswith('COLUMNS'):
                section = 'COLUMNS'
            elif line.startswith('RHS'):
                section = 'RHS'
            elif line.startswith('BOUNDS'):
                section = 'BOUNDS'
            elif line.startswith('ENDATA'):
                break
            
            if section == 'ROWS':
                self.process_rows(line)
            elif section == 'COLUMNS':
                self.process_columns(line)
            elif section == 'RHS':
                self.process_rhs(line)
            elif section == 'BOUNDS':
                self.process_bounds(line)

    def process_rows(self, line):
        if line.startswith('N'):
            return  # Ignorar linhas de nome
        elif line.startswith('G'):
            self.A_ub.append([])  # Adicionar uma nova linha para A_ub
            self.b_ub.append(0)  # Inicializar b_ub
        elif line.startswith('L'):
            self.A_ub.append([])  # Adicionar uma nova linha para A_ub
            self.b_ub.append(0)  # Inicializar b_ub
        elif line.startswith('E'):
            self.A_eq.append([])  # Adicionar uma nova linha para A_eq
            self.b_eq.append(0)  # Inicializar b_eq

    def process_columns(self, line):
        parts = line.split()
        if len(parts) < 2:
            return  # Ignorar linhas que não têm o formato esperado
        var_name = parts[0]
        if var_name not in self.var_names:
            self.var_names.append(var_name)
            self.c = np.zeros(len(self.var_names))  # Inicializa o vetor de coeficientes

        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):  # Verifica se há um coeficiente correspondente
                print(f"Coeficiente faltando na linha de colunas: {line}")
                continue
            row_name = parts[i]
            coeff = float(parts[i + 1])
            if row_name.startswith('G') or row_name.startswith('L'):
                idx = len(self.A_ub) - 1  # Última linha adicionada
                self.A_ub[idx].append(coeff)
            elif row_name.startswith('E'):
                idx = len(self.A_eq) - 1  # Última linha adicionada
                self.A_eq[idx].append(coeff)

    def process_rhs(self, line):
        parts = line.split()
        if len(parts) < 2:
            return  # Ignorar linhas que não têm o formato esperado
        row_name = parts[0]
        try:
            rhs_value = float(parts[1])
        except ValueError:
            print(f"Valor inválido na linha RHS: {line}")
            return

        if row_name.startswith('G') or row_name.startswith('L'):
            idx = len(self.b_ub) - 1  # Última linha adicionada
            self.b_ub[idx] = rhs_value
        elif row_name.startswith('E'):
            idx = len(self.b_eq) - 1  # Última linha adicionada
            self.b_eq[idx] = rhs_value

    def process_bounds(self, line):
        parts = line.split()
        var_name = parts[0]
        bound_type = parts[1]
        bound_value = float(parts[2])
        idx = self.var_names.index(var_name)

        if bound_type == 'LO':
            self.bounds.append((bound_value, None))  # Limite inferior
        elif bound_type == 'UP':
            self.bounds.append((None, bound_value))  # Limite superior
        elif bound_type == 'FX':
            self.bounds.append((bound_value, bound_value))  # Valor fixo

    def to_linprog_format(self):
        return {
            'c': self.c,
            'A_ub': np.array(self.A_ub) if self.A_ub else None,
            'b_ub': np.array(self.b_ub) if self.b_ub else None,
            'A_eq': np.array(self.A_eq) if self.A_eq else None,
            'b_eq': np.array(self.b_eq) if self.b_eq else None,
            'bounds': self.bounds if self.bounds else None
        }

# Exemplo de uso
if __name__ == "__main__":
    mps_file_path = '/home/vaio/ufpb/Large-Scale-Optimization/Instancias/25fv47.mps'
    reader = MPSReader(mps_file_path)
    reader.read_mps()
    linprog_data = reader.to_linprog_format()

    # Exibir os dados convertidos
    print("Coeficientes da função objetivo (c):", linprog_data['c'])
    print("Matriz de restrições de desigualdade (A_ub):", linprog_data['A_ub'])
    print("Vetor de limites de desigualdade (b_ub):", linprog_data['b_ub'])
    print("Matriz de restrições de igualdade (A_eq):", linprog_data['A_eq'])
    print("Vetor de limites de igualdade (b_eq):", linprog_data['b_eq'])
    print("Limites das variáveis (bounds):", linprog_data['bounds'])