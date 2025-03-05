import re
import numpy as np

class MPSParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = ""
        self.rows = []
        self.A = {}
        self.rhs = {}
    
    def extract_name(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith("NAME"):
                    self.name = line.split()[1]
                    break
        return self.name
    


    def extract_rows(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("ROWS\n") + 1
            end = lines.index("COLUMNS\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) == 2:
                    self.rows.append(parts[1])
        return self.rows
    
    def extract_columns(self):
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
    
    def extract_rhs(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            start = lines.index("RHS\n") + 1
            end = lines.index("ENDATA\n")
            
            for line in lines[start:end]:
                parts = line.split()
                if len(parts) >= 3:
                    _, row_name, value = parts[:3]
                    value = float(value)
                    self.rhs[row_name] = value
                    
                    if len(parts) == 5:
                        row_name2, value2 = parts[3:]
                        value2 = float(value2)
                        self.rhs[row_name2] = value2
        return self.rhs
    
    def parse(self):
        self.extract_name()
        self.extract_rows()
        self.extract_columns()
        self.extract_rhs()
        return {
            "name": self.name,
            "rows": self.rows,
            "A": self.A,
            "rhs": self.rhs
        }

if __name__ == "__main__":
    
    # Exemplo de uso:
    parser = MPSParser("/home/vaio/ufpb/Large-Scale-Optimization/Instancias/25fv47.mps")
    data = parser.parse()
    print("\nName",data["name"])
    

