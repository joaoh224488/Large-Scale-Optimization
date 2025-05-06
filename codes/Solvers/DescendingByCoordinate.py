import re
import os
import time
import random
import pandas as pd
import numpy as np
import logging
from scipy.sparse import csc_matrix
from scipy.optimize import linprog

class CoordinateDescent:
    """
    Classe para resolver problemas de mínimos quadrados lineares usando diferentes métodos de descida por coordenadas.

    Atributos:
        filepath (str): Caminho para o arquivo de dados
        data (dict): Dados do problema extraídos do arquivo
        A (csc_matrix): Matriz esparsa do problema
        b (np.array): Vetor de termos independentes
        results (dict): Armazena os resultados dos diferentes métodos

    Métodos:
        parse_file(): Faz o parsing do arquivo de entrada
        run(): Executa todos os métodos de otimização
        print_results(): Imprime os resultados comparativos
        get_results(): Retorna os resultados em formato de dicionário
        _calculate_objective(): Calcula o valor da função objetivo
        _check_convergence(): Verifica critérios de convergência
        solve_ccd(): Método de Descida Cíclica por Coordenadas
        solve_rcd(): Método de Descida Aleatória por Coordenadas
        solve_mdcd(): Método de Descida por Coordenadas de Máxima Descida
    """

    def __init__(self, filepath):
        """
        Inicializa o solver com o caminho do arquivo de dados.

        Args:
            filepath (str): Caminho para o arquivo .txt com os dados do problema
        """
        self.filepath = filepath
        self.data = None
        self.A = None
        self.b = None
        self.results = {}

    def parse_file(self):
        """Faz o parsing do arquivo de entrada e prepara os dados do problema."""
        data = {"row": [], "col": [], "data": [], "b": []}
        current_key = None

        try:
            with open(self.filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("matriz.row :"):
                        current_key = "row"
                        line = line.replace("matriz.row :", "").strip()
                    elif line.startswith("matriz.col :"):
                        current_key = "col"
                        line = line.replace("matriz.col :", "").strip()
                    elif line.startswith("matriz.Data :"):
                        current_key = "data"
                        line = line.replace("matriz.Data :", "").strip()
                    elif line.startswith("Vector b :"):
                        current_key = "b"
                        line = line.replace("Vector b :", "").strip()
                    elif current_key is None:
                        continue

                    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if not numbers:
                        continue

                    if current_key in ["row", "col"]:
                        data[current_key].extend([int(num) for num in numbers])
                    elif current_key in ["data", "b"]:
                        data[current_key].extend([float(num) for num in numbers])

            # Verifica dados mínimos
            if not all(data.values()):
                missing_keys = [k for k, v in data.items() if not v]
                raise ValueError(f"Dados insuficientes no arquivo. Chaves faltando: {missing_keys}")

            # Converte para arrays numpy
            for key in data:
                if key in ["row", "col"]:
                    data[key] = np.array(data[key], dtype=int)
                else:
                    data[key] = np.array(data[key], dtype=float)

            # Valida consistência dos dados
            if len(data['row']) != len(data['col']) or len(data['row']) != len(data['data']):
                raise ValueError("Inconsistência nos tamanhos de matriz.row :, matriz.col : e matriz.Data :.")
            if data['row'].min() < 0 or data['col'].min() < 0:
                raise ValueError("Índices de linha/coluna negativos encontrados.")

            # Cria matriz esparsa
            m = data['b'].shape[0]
            n = data['col'].max() + 1
            self.A = csc_matrix((data['data'], (data['row'], data['col'])), shape=(m, n))
            self.b = data['b']
            self.data = data

            print(f"Arquivo {self.filepath} processado com sucesso.")
            print(f"Matriz A criada com shape {self.A.shape} ({self.A.nnz} nnz) e vetor b com shape {self.b.shape}.")
            return True

        except Exception as e:
            logging.error(f"Erro no parsing do arquivo: {e}")
            return False

    def _calculate_objective(self, x):
        """Calcula o valor da função objetivo 0.5 * ||Ax - b||^2."""
        residual = self.A.dot(x) - self.b
        return 0.5 * np.linalg.norm(residual)**2

    def _check_convergence(self, iter_count, max_iter, r, tolerancia):
        """Verifica a convergência com base na norma do gradiente."""
        grad_norm = np.linalg.norm(self.A.T @ r)
        converged = grad_norm < tolerancia
        if converged:
            print(f"Convergência atingida na iteração {iter_count} (||A^T*r|| = {grad_norm:.2e} < {tolerancia:.2e})")
        elif iter_count >= max_iter:
            print(f"Máximo de iterações ({max_iter}) atingido. Norma do gradiente final: {grad_norm:.2e}")
        return converged

    def solve_ccd(self, max_iter=10000, tol=1e-6, verbose=True):
        """Método de Descida Cíclica por Coordenadas (CCD)."""
        m, n = self.A.shape
        x = np.zeros(n)
        r = self.A.dot(x) - self.b
        norma_colunas_A2 = np.array([self.A.getcol(j).power(2).sum() for j in range(n)])

        colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
        if len(colunas_nulas) > 0 and verbose:
            print(f"Aviso (CCD): Colunas {colunas_nulas} têm norma quase nula. Serão ignoradas.")
            norma_colunas_A2[colunas_nulas] = np.inf

        if verbose:
            print("\n--- Iniciando Descida Cíclica por Coordenadas (CCD) ---")
        
        tempo_inicio = time.time()
        iter_count = 0
        converged = False

        for epoca in range(max_iter // n + 1):
            x_anterior_epoca = x.copy()
            tempo_epoca = time.time()

            for j in range(n):
                if norma_colunas_A2[j] == np.inf:
                    continue

                grad_j = self.A.getcol(j).T.dot(r).item()
                delta_x_j = -grad_j / norma_colunas_A2[j]

                if abs(delta_x_j) > 1e-15:
                    x[j] += delta_x_j
                    r += self.A.getcol(j).multiply(delta_x_j).toarray().flatten()

                iter_count += 1
                if iter_count % n == 0:
                    if self._check_convergence(iter_count, max_iter, r, tol):
                        converged = True
                        break
                if iter_count >= max_iter:
                    break

            if converged or iter_count >= max_iter:
                break

            if verbose and (epoca % 10 == 0 or epoca == (max_iter // n)):
                delta_x_epoca = np.linalg.norm(x - x_anterior_epoca)
                f_x_atual = 0.5 * np.linalg.norm(r)**2
                tempo_passado_epoca = time.time() - tempo_epoca
                print(f"CCD Época {epoca+1}, Iter {iter_count}, f(x): {f_x_atual:.6e}, ||Δx_epoca||: {delta_x_epoca:.6e}, Tempo Época: {tempo_passado_epoca:.2f}s")

        tempo_total = time.time() - tempo_inicio
        f_x_final = self._calculate_objective(x)
        
        if verbose:
            print(f"CCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
            if not converged and iter_count >= max_iter:
                grad_norm = np.linalg.norm(self.A.T @ r)
                print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
            print(f"CCD f(x*) = {f_x_final:.6e}")
        
        return x, f_x_final, iter_count, tempo_total

    def solve_rcd(self, max_iter=10000, tol=1e-6, verbose=True):
        """Método de Descida Aleatória por Coordenadas (RCD)."""
        m, n = self.A.shape
        x = np.zeros(n)
        r = self.A.dot(x) - self.b
        norma_colunas_A2 = np.array([self.A.getcol(j).power(2).sum() for j in range(n)])

        colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
        indices_validos = [j for j in range(n) if norma_colunas_A2[j] >= 1e-15]
        
        if len(colunas_nulas) > 0 and verbose:
            print(f"Aviso (RCD): Colunas {colunas_nulas} têm norma quase nula. Não serão selecionadas.")
        
        if not indices_validos:
            raise ValueError("Todas as colunas têm norma nula. Impossível prosseguir.")

        if verbose:
            print("\n--- Iniciando Descida Aleatória por Coordenadas (RCD) ---")
        
        tempo_inicio = time.time()
        converged = False

        for iter_count in range(1, max_iter + 1):
            j = random.choice(indices_validos)
            grad_j = self.A.getcol(j).T.dot(r).item()
            delta_x_j = -grad_j / norma_colunas_A2[j]

            if abs(delta_x_j) > 1e-15:
                x[j] += delta_x_j
                r += self.A.getcol(j).multiply(delta_x_j).toarray().flatten()

            if iter_count % n == 0:
                if self._check_convergence(iter_count, max_iter, r, tol):
                    converged = True
                    break
                elif verbose and (iter_count % (n * 10) == 0 or iter_count == max_iter):
                    f_x_atual = 0.5 * np.linalg.norm(r)**2
                    grad_norm = np.linalg.norm(self.A.T @ r)
                    print(f"RCD Iter {iter_count}/{max_iter}, f(x): {f_x_atual:.6e}, ||∇f(x)||: {grad_norm:.2e}")

        tempo_total = time.time() - tempo_inicio
        f_x_final = self._calculate_objective(x)
        
        if verbose:
            print(f"RCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
            if not converged and iter_count >= max_iter:
                grad_norm = np.linalg.norm(self.A.T @ r)
                print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
            print(f"RCD f(x*) = {f_x_final:.6e}")
        
        return x, f_x_final, iter_count, tempo_total

    def solve_mdcd(self, max_iter=10000, tol=1e-6, verbose=True):
        """Método de Descida por Coordenadas de Máxima Descida (MDCD)."""
        m, n = self.A.shape
        x = np.zeros(n)
        r = self.A.dot(x) - self.b
        norma_colunas_A2 = np.array([self.A.getcol(j).power(2).sum() for j in range(n)])

        colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
        if len(colunas_nulas) > 0 and verbose:
            print(f"Aviso (MDCD): Colunas {colunas_nulas} têm norma quase nula. Serão ignoradas.")
            norma_colunas_A2[colunas_nulas] = np.inf

        if verbose:
            print("\n--- Iniciando Descida por Coordenadas de Máxima Descida (MDCD) ---")
        
        tempo_inicio = time.time()
        converged = False

        for iter_count in range(1, max_iter + 1):
            grad = self.A.T @ r
            potencial_descida = np.zeros(n)
            valid_indices = norma_colunas_A2 != np.inf
            potencial_descida[valid_indices] = (grad[valid_indices]**2) / norma_colunas_A2[valid_indices]

            if np.all(potencial_descida < 1e-15):
                if verbose:
                    print(f"MDCD: Nenhuma coordenada oferece descida significativa na iteração {iter_count}.")
                if self._check_convergence(iter_count, max_iter, r, tol):
                    converged = True
                break

            j_star = np.argmax(potencial_descida)
            grad_j = grad[j_star]
            delta_x_j = -grad_j / norma_colunas_A2[j_star]

            if abs(delta_x_j) > 1e-15:
                x[j_star] += delta_x_j
                r += self.A.getcol(j_star).multiply(delta_x_j).toarray().flatten()

            if iter_count % n == 0:
                if self._check_convergence(iter_count, max_iter, r, tol):
                    converged = True
                    break
                elif verbose and (iter_count % (n * 10) == 0 or iter_count == max_iter):
                    f_x_atual = 0.5 * np.linalg.norm(r)**2
                    grad_norm = np.linalg.norm(self.A.T @ r)
                    print(f"MDCD Iter {iter_count}/{max_iter}, f(x): {f_x_atual:.6e}, ||∇f(x)||: {grad_norm:.2e}, MaxDesc: {potencial_descida[j_star]:.2e}")

        tempo_total = time.time() - tempo_inicio
        f_x_final = self._calculate_objective(x)
        
        if verbose:
            print(f"MDCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
            if not converged and iter_count >= max_iter:
                grad_norm = np.linalg.norm(self.A.T @ r)
                print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
            print(f"MDCD f(x*) = {f_x_final:.6e}")
        
        return x, f_x_final, iter_count, tempo_total

    def run(self, max_iter=10000, tol=1e-6):
        """Executa todos os métodos de otimização e armazena os resultados."""
        if not self.parse_file():
            return False

        try:
            # Resolve com CCD
            x_ccd, f_ccd, it_ccd, t_ccd = self.solve_ccd(max_iter, tol)
            self.results['ccd'] = {'x': x_ccd, 'f': f_ccd, 'iter': it_ccd, 'time': t_ccd}
            np.savez('lsq_results_ccd.npz', x_star=x_ccd, f_x_star=f_ccd, iterations=it_ccd, time_sec=t_ccd)
            print("Resultado CCD salvo em lsq_results_ccd.npz")

            # Resolve com RCD
            x_rcd, f_rcd, it_rcd, t_rcd = self.solve_rcd(max_iter, tol)
            self.results['rcd'] = {'x': x_rcd, 'f': f_rcd, 'iter': it_rcd, 'time': t_rcd}
            np.savez('lsq_results_rcd.npz', x_star=x_rcd, f_x_star=f_rcd, iterations=it_rcd, time_sec=t_rcd)
            print("Resultado RCD salvo em lsq_results_rcd.npz")

            # Resolve com MDCD
            x_mdcd, f_mdcd, it_mdcd, t_mdcd = self.solve_mdcd(max_iter, tol)
            self.results['mdcd'] = {'x': x_mdcd, 'f': f_mdcd, 'iter': it_mdcd, 'time': t_mdcd}
            np.savez('lsq_results_mdcd.npz', x_star=x_mdcd, f_x_star=f_mdcd, iterations=it_mdcd, time_sec=t_mdcd)
            print("Resultado MDCD salvo em lsq_results_mdcd.npz")

            return True

        except Exception as e:
            logging.error(f"Erro durante a execução dos métodos: {e}")
            return False

    def print_results(self):
        """Imprime um resumo comparativo dos resultados."""
        if not self.results:
            print("Nenhum resultado disponível. Execute o método run() primeiro.")
            return

        print("\n--- Resumo dos Resultados ---")
        for method, res in self.results.items():
            print(f"Método: {method.upper()}")
            print(f"  Iterações: {res['iter']}")
            print(f"  Tempo (s): {res['time']:.4f}")
            print(f"  f(x*): {res['f']:.6e}")
            print("  -------------------------")

    def get_results(self):
        """
        Retorna os resultados detalhados da otimização em formato de dicionário.
        
        Returns:
            dict: Dicionário contendo:
                - Para cada método (CCD, RCD, MDCD):
                    - "SOLUCAO_X": Vetor solução encontrado
                    - "VALOR_OBJETIVO": Valor da função objetivo na solução
                    - "ITERACOES": Número de iterações realizadas
                    - "TEMPO (s)": Tempo de execução em segundos
                    - "NORM_GRADIENTE": Norma do gradiente na solução (medida de convergência)
                - "MELHOR_METODO": Nome do método com menor valor objetivo
                - "COMPARACAO": DataFrame comparando os métodos
                None: Se nenhum resultado estiver disponível
        """
        if not self.results:
            return None
        
        try:
            # Calcula métricas adicionais para cada método
            for method, res in self.results.items():
                x = res['x']
                r = self.A.dot(x) - self.b  # Residual
                res['norm_grad'] = np.linalg.norm(self.A.T @ r)  # Norma do gradiente
                res['norm_x'] = np.linalg.norm(x)  # Norma da solução
            
            # Identifica o melhor método (menor valor objetivo)
            melhor_metodo = min(self.results.keys(), 
                            key=lambda k: self.results[k]['f'])
            
            # Cria DataFrame comparativo
            df_comparacao = pd.DataFrame({
                'Método': list(self.results.keys()),
                'Valor Objetivo': [res['f'] for res in self.results.values()],
                'Iterações': [res['iter'] for res in self.results.values()],
                'Tempo (s)': [res['time'] for res in self.results.values()],
                'Norma do Gradiente': [res['norm_grad'] for res in self.results.values()],
                'Norma da Solução': [res['norm_x'] for res in self.results.values()]
            }).set_index('Método')
            
            # Organiza os resultados no formato de dicionário
            resultados = {
                "MELHOR_METODO": melhor_metodo.upper(),
                "COMPARACAO": df_comparacao
            }
            
            # Adiciona resultados detalhados por método
            for method, res in self.results.items():
                resultados[method.upper()] = {
                    "SOLUCAO_X": res['x'],
                    "VALOR_OBJETIVO": res['f'],
                    "ITERACOES": res['iter'],
                    "TEMPO (s)": res['time'],
                    "NORM_GRADIENTE": res['norm_grad'],
                    "NORM_SOLUCAO": res['norm_x']
                }
            
            return resultados
        
        except Exception as e:
            logging.error(f"Erro ao processar os resultados: {e}")
            return None
    


def main():
    """Função principal para execução via linha de comando."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python lsq_solver.py arquivo.txt [max_iter] [tolerancia]")
        print("Exemplo: python lsq_solver.py lsq_data.txt 10000 1e-6")
        # python3 coordenate_descent.py lsq_1000_1e-04_1.txt 1000
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6

    if not os.path.exists(input_file):
        print(f"Erro: Arquivo '{input_file}' não encontrado.")
        sys.exit(1)

    solver = CoordinateDescent(input_file)
    if solver.run(max_iter, tol):
        solver.print_results()


if __name__ == "__main__":
    main()
