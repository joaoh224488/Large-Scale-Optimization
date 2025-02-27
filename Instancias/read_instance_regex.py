import numpy as np

class MPSParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = ""
        self.rows = {}
        self.columns = {}
        self.rhs = {}
        self.bounds = {}

    def extract_name(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith("NAME"):
                    self.name = line.split()[1]
                    break

    def extract_rows(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("ROWS\n") + 1
            end = lines.index("COLUMNS\n")

            for line in lines[start:end]:
                parts = line.strip().split()
                if len(parts) == 2:
                    row_type, row_name = parts
                    self.rows[row_name] = row_type  # 'N' (obj), 'E' (=), 'L' (≤), 'G' (≥)

    def extract_columns(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("COLUMNS\n") + 1
            end = lines.index("RHS\n")

            for line in lines[start:end]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    var_name = parts[0]
                    for i in range(1, len(parts), 2):
                        row_name, coeff = parts[i], float(parts[i+1])
                        if var_name not in self.columns:
                            self.columns[var_name] = {}
                        self.columns[var_name][row_name] = coeff

    def extract_rhs(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("RHS\n") + 1
            try:
                end = lines.index("BOUNDS\n")
            except ValueError:
                end = lines.index("ENDATA\n")

            for line in lines[start:end]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    rhs_name = parts[0]
                    for i in range(1, len(parts), 2):
                        row_name, value = parts[i], float(parts[i+1])
                        self.rhs[row_name] = value

    def parse(self):
        self.extract_name()
        self.extract_rows()
        self.extract_columns()
        self.extract_rhs()

        variables = sorted(self.columns.keys())
        num_vars = len(variables)

        # Função objetivo
        c = np.array([self.columns[v].get("R0000", 0) for v in variables])

        # Restrições de igualdade
        A_eq = []
        b_eq = []

        # Restrições de desigualdade
        A_ub = []
        b_ub = []

        for row_name, row_type in self.rows.items():
            if row_name == "R0000":
                continue  # Ignorar a função objetivo

            row_vector = np.array([self.columns[v].get(row_name, 0) for v in variables])
            rhs_value = self.rhs.get(row_name, 0)

            if row_type == "E":
                A_eq.append(row_vector)
                b_eq.append(rhs_value)
            elif row_type == "L":
                A_ub.append(row_vector)
                b_ub.append(rhs_value)
            elif row_type == "G":
                A_ub.append(-row_vector)
                b_ub.append(-rhs_value)

        # Transformar listas em arrays do NumPy
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        # Limites das variáveis (assumindo todas >= 0 se não houver BOUNDS)
        bounds = [(0, None) for _ in variables]

        return {
            "name": self.name,
            "c": c,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "bounds": bounds
        }

# Exemplo de uso:
if __name__ == "__main__":
   pass