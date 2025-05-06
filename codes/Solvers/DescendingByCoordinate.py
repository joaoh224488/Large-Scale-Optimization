import re
import os
import time
import random
import numpy as np

from scipy.sparse import csc_matrix, linalg as splinalg

# --- Função para parsing do arquivo LSQ ---
def parse_lsq_file(filepath):
    """Lê um arquivo .txt no formato especificado e extrai dados da matriz A e vetor b."""
    data = {"row": [], "col": [], "data": [], "b": []}
    current_key = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("A.row:"):
                    current_key = "row"
                    line = line.replace("A.row:", "").strip()
                elif line.startswith("A.col:"):
                    current_key = "col"
                    line = line.replace("A.col:", "").strip()
                elif line.startswith("A.data:"):
                    current_key = "data"
                    line = line.replace("A.data:", "").strip()
                elif line.startswith("b:"):
                    current_key = "b"
                    line = line.replace("b:", "").strip()
                # Ignora linhas que não começam com os marcadores esperados após definir uma chave
                elif current_key is None:
                    continue

                # Extrai números da linha atual
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if not numbers:
                    continue # Pula se não encontrar números na linha

                if current_key in ["row", "col"]:
                    try:
                        data[current_key].extend([int(num) for num in numbers])
                    except ValueError as e:
                        print(f"Aviso: Erro ao converter para int em {current_key} na linha '{line.strip()}': {e}. Ignorando valor.")
                elif current_key in ["data", "b"]:
                    try:
                        data[current_key].extend([float(num) for num in numbers])
                    except ValueError as e:
                        print(f"Aviso: Erro ao converter para float em {current_key} na linha '{line.strip()}': {e}. Ignorando valor.")

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {filepath}")
        return None
    except Exception as e:
        print(f"Erro inesperado ao ler o arquivo {filepath}: {e}")
        return None

    # Verifica se todos os dados necessários foram lidos
    if not all(data.values()):
        print("Erro: Faltando dados no arquivo (A.row, A.col, A.data ou b).")
        # Verifica quais chaves estão faltando
        missing_keys = [k for k, v in data.items() if not v]
        if missing_keys:
            print(f"Chaves com dados ausentes: {missing_keys}")
        return None

    # Converte listas para arrays numpy
    try:
        for key in data:
            if key in ["row", "col"]:
                data[key] = np.array(data[key], dtype=int)
            else:
                data[key] = np.array(data[key], dtype=float)
    except Exception as e:
        print(f"Erro ao converter listas para arrays numpy: {e}")
        return None

    # Validações adicionais
    if len(data['row']) != len(data['col']) or len(data['row']) != len(data['data']):
        print("Erro: Inconsistência nos tamanhos de A.row, A.col e A.data.")
        return None
    if data['row'].min() < 0 or data['col'].min() < 0:
        print("Erro: Índices de linha/coluna negativos encontrados (esperado base 1 no arquivo).")
        return None

    print(f"Arquivo {filepath} processado com sucesso.")
    return data

# --- Funções de Resolução LSQ ---

def _calculate_objective(A, x, b):
    """Calcula o valor da função objetivo 0.5 * ||Ax - b||^2."""
    residual = A.dot(x) - b
    return 0.5 * np.linalg.norm(residual)**2

def _check_convergence(iter_count, max_iter, A, r, tolerancia):
     """Verifica a convergência com base na norma do gradiente."""
     # O gradiente é A.T @ (Ax - b) = A.T @ r
     grad_norm = np.linalg.norm(A.T @ r)
     converged = grad_norm < tolerancia
     if converged:
         print(f"Convergência atingida na iteração {iter_count} (||A^T*r|| = {grad_norm:.2e} < {tolerancia:.2e})")
     elif iter_count >= max_iter:
         print(f"Máximo de iterações ({max_iter}) atingido. Norma do gradiente final: {grad_norm:.2e}")
     return converged

def solve_ccd(A, b, max_iteracoes=10000, tolerancia=1e-6, verbose=True):
    """Resolve min 0.5*||Ax-b||^2 usando Descida Cíclica por Coordenadas (CCD)."""
    m, n = A.shape
    x = np.zeros(n)
    r = A.dot(x) - b # Residual atual: Ax - b
    norma_colunas_A2 = np.array([A.getcol(j).power(2).sum() for j in range(n)])

    # Lida com colunas nulas para evitar divisão por zero
    colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
    if len(colunas_nulas) > 0:
        if verbose: print(f"Aviso (CCD): Colunas {colunas_nulas} têm norma quase nula. Serão ignoradas nas atualizações.")
        norma_colunas_A2[colunas_nulas] = np.inf # Evita atualização

    if verbose: print("\n--- Iniciando Descida Cíclica por Coordenadas (CCD) ---")
    tempo_inicio = time.time()
    iter_count = 0
    converged = False

    for epoca in range(max_iteracoes // n + 1):
        x_anterior_epoca = x.copy()
        tempo_epoca = time.time()

        for j in range(n):
            if norma_colunas_A2[j] == np.inf: continue # Pula colunas nulas

            # Gradiente parcial: g_j = (A.T @ r)_j = A_j.T @ r
            grad_j = A.getcol(j).T.dot(r).item()

            # Calcula a mudança ótima para a coordenada j
            delta_x_j = -grad_j / norma_colunas_A2[j]

            # Atualiza x e o residual r
            if abs(delta_x_j) > 1e-15: # Evita atualizações desnecessárias
                x[j] += delta_x_j
                # Atualização eficiente do residual: r = r + A_j * delta_x_j
                r += A.getcol(j).multiply(delta_x_j).toarray().flatten() # Multiplica coluna esparsa por escalar e achata

            iter_count += 1
            # Verificação de convergência a cada N iterações (uma época)
            if iter_count % n == 0:
                 if _check_convergence(iter_count, max_iteracoes, A, r, tolerancia):
                     converged = True
                     break
            if iter_count >= max_iteracoes: break

        if converged or iter_count >= max_iteracoes: break

        # Log da época (opcional)
        delta_x_epoca = np.linalg.norm(x - x_anterior_epoca)
        f_x_atual = 0.5 * np.linalg.norm(r)**2
        tempo_passado_epoca = time.time() - tempo_epoca
        if verbose and (epoca % 10 == 0 or epoca == (max_iteracoes // n)): # Log a cada 10 épocas
             print(f"CCD Época {epoca+1}, Iter {iter_count}, f(x): {f_x_atual:.6e}, ||Δx_epoca||: {delta_x_epoca:.6e}, Tempo Época: {tempo_passado_epoca:.2f}s")

    tempo_total = time.time() - tempo_inicio
    f_x_final = _calculate_objective(A, x, b)
    if verbose:
        print(f"CCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
        if not converged and iter_count >= max_iteracoes:
             grad_norm = np.linalg.norm(A.T @ r)
             print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
        print(f"CCD f(x*) = {f_x_final:.6e}")
    return x, f_x_final, iter_count, tempo_total

def solve_rcd(A, b, max_iteracoes=10000, tolerancia=1e-6, verbose=True):
    """Resolve min 0.5*||Ax-b||^2 usando Descida Aleatória por Coordenadas (RCD)."""
    m, n = A.shape
    x = np.zeros(n)
    r = A.dot(x) - b # Residual atual: Ax - b
    norma_colunas_A2 = np.array([A.getcol(j).power(2).sum() for j in range(n)])

    colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
    indices_validos = [j for j in range(n) if norma_colunas_A2[j] >= 1e-15]
    if len(colunas_nulas) > 0:
        if verbose: print(f"Aviso (RCD): Colunas {colunas_nulas} têm norma quase nula. Não serão selecionadas.")
        if not indices_validos:
             print("Erro (RCD): Todas as colunas têm norma nula. Impossível prosseguir.")
             return x, 0.5 * np.linalg.norm(r)**2, 0, 0.0
        norma_colunas_A2[colunas_nulas] = np.inf # Marca para referência, mas não serão usadas

    if verbose: print("\n--- Iniciando Descida Aleatória por Coordenadas (RCD) ---")
    tempo_inicio = time.time()
    converged = False

    for iter_count in range(1, max_iteracoes + 1):
        # Seleciona uma coordenada válida aleatoriamente (uniforme)
        j = random.choice(indices_validos)

        # Gradiente parcial: g_j = A_j.T @ r
        grad_j = A.getcol(j).T.dot(r).item()

        # Calcula a mudança ótima para a coordenada j
        delta_x_j = -grad_j / norma_colunas_A2[j]

        # Atualiza x e o residual r
        if abs(delta_x_j) > 1e-15:
            x[j] += delta_x_j
            r += A.getcol(j).multiply(delta_x_j).toarray().flatten()

        # Verifica convergência periodicamente (e.g., a cada n iterações)
        if iter_count % n == 0:
            if _check_convergence(iter_count, max_iteracoes, A, r, tolerancia):
                converged = True
                break
            # Log periódico (opcional)
            elif verbose and (iter_count % (n * 10) == 0 or iter_count == max_iteracoes):
                 f_x_atual = 0.5 * np.linalg.norm(r)**2
                 grad_norm = np.linalg.norm(A.T @ r)
                 print(f"RCD Iter {iter_count}/{max_iteracoes}, f(x): {f_x_atual:.6e}, ||∇f(x)||: {grad_norm:.2e}")

    tempo_total = time.time() - tempo_inicio
    f_x_final = _calculate_objective(A, x, b)
    if verbose:
        print(f"RCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
        if not converged and iter_count >= max_iteracoes:
             grad_norm = np.linalg.norm(A.T @ r)
             print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
        print(f"RCD f(x*) = {f_x_final:.6e}")
    return x, f_x_final, iter_count, tempo_total

def solve_mdcd(A, b, max_iteracoes=10000, tolerancia=1e-6, verbose=True):
    """Resolve min 0.5*||Ax-b||^2 usando Descida por Coordenadas de Máxima Descida (MDCD)."""
    m, n = A.shape
    x = np.zeros(n)
    r = A.dot(x) - b # Residual atual: Ax - b
    norma_colunas_A2 = np.array([A.getcol(j).power(2).sum() for j in range(n)])

    colunas_nulas = np.where(norma_colunas_A2 < 1e-15)[0]
    if len(colunas_nulas) > 0:
        if verbose: print(f"Aviso (MDCD): Colunas {colunas_nulas} têm norma quase nula. Serão ignoradas.")
        norma_colunas_A2[colunas_nulas] = np.inf # Evita seleção e divisão por zero

    if verbose: print("\n--- Iniciando Descida por Coordenadas de Máxima Descida (MDCD) ---")
    tempo_inicio = time.time()
    converged = False

    for iter_count in range(1, max_iteracoes + 1):
        # Calcula o gradiente completo projetado nas coordenadas
        grad = A.T @ r

        # Calcula o potencial de descida para cada coordenada (ignora colunas nulas)
        # Descida ~ (grad_j)^2 / ||A_j||^2
        # Usamos valor absoluto do gradiente para evitar problemas com np.argmax em valores negativos
        # O passo ótimo é -grad_j / ||A_j||^2, a descida é proporcional a grad_j^2 / ||A_j||^2
        potencial_descida = np.zeros(n)
        valid_indices = norma_colunas_A2 != np.inf
        potencial_descida[valid_indices] = (grad[valid_indices]**2) / norma_colunas_A2[valid_indices]

        # Encontra a coordenada com máxima descida
        if np.all(potencial_descida < 1e-15): # Se não há descida significativa
             if verbose: print(f"MDCD: Nenhuma coordenada oferece descida significativa na iteração {iter_count}.")
             # Verifica convergência final pelo gradiente
             if _check_convergence(iter_count, max_iteracoes, A, r, tolerancia):
                 converged = True
             break

        j_star = np.argmax(potencial_descida)

        # Calcula a mudança ótima para a coordenada j_star
        grad_j = grad[j_star]
        delta_x_j = -grad_j / norma_colunas_A2[j_star]

        # Atualiza x e o residual r
        if abs(delta_x_j) > 1e-15:
            x[j_star] += delta_x_j
            r += A.getcol(j_star).multiply(delta_x_j).toarray().flatten()

        # Verifica convergência periodicamente (e.g., a cada n iterações ou pela descida máxima)
        # Usar a norma do gradiente é mais robusto
        if iter_count % n == 0: # Checa a cada N iterações
            if _check_convergence(iter_count, max_iteracoes, A, r, tolerancia):
                converged = True
                break
            # Log periódico (opcional)
            elif verbose and (iter_count % (n * 10) == 0 or iter_count == max_iteracoes):
                 f_x_atual = 0.5 * np.linalg.norm(r)**2
                 grad_norm = np.linalg.norm(A.T @ r)
                 print(f"MDCD Iter {iter_count}/{max_iteracoes}, f(x): {f_x_atual:.6e}, ||∇f(x)||: {grad_norm:.2e}, MaxDesc: {potencial_descida[j_star]:.2e}")

    tempo_total = time.time() - tempo_inicio
    f_x_final = _calculate_objective(A, x, b)
    if verbose:
        print(f"MDCD Finalizado em {tempo_total:.2f}s ({iter_count} iterações).")
        if not converged and iter_count >= max_iteracoes:
             grad_norm = np.linalg.norm(A.T @ r)
             print(f"Máximo de iterações atingido. Norma do Gradiente Final: {grad_norm:.2e}")
        print(f"MDCD f(x*) = {f_x_final:.6e}")
    return x, f_x_final, iter_count, tempo_total

# --- Pipeline principal ---
def main(txt_filename, max_iter=10000, tol=1e-6):
    # 1. Parse do arquivo
    print(f"Processando arquivo: {txt_filename}")
    dados = parse_lsq_file(txt_filename)
    if dados is None:
        print("Falha no parsing do arquivo. Abortando.")
        return

    # 2. Extrai dados e cria matriz esparsa A
    try:
        row, col, data, b = dados['row'], dados['col'], dados['data'], dados['b']
        m = b.shape[0]
        # Determina n como o maior índice de coluna + 1 (considerando base 0)
        n = col.max() + 1
        A = csc_matrix((data, (row, col)), shape=(m, n))
        print(f"Matriz A criada com shape {A.shape} ({A.nnz} nnz) e vetor b com shape {b.shape}.")
    except Exception as e:
        print(f"Erro ao criar a matriz esparsa A: {e}")
        return

    # --- Execução dos Solvers ---
    results = {}

    # 3.1 Resolve com CCD
    try:
        x_ccd, f_ccd, it_ccd, t_ccd = solve_ccd(A, b, max_iteracoes=max_iter, tolerancia=tol)
        results['ccd'] = {'x': x_ccd, 'f': f_ccd, 'iter': it_ccd, 'time': t_ccd}
        np.savez('lsq_results_ccd.npz', x_star=x_ccd, f_x_star=f_ccd, iterations=it_ccd, time_sec=t_ccd)
        print("Resultado CCD salvo em lsq_results_ccd.npz")
    except Exception as e:
        print(f"Erro ao executar CCD: {e}")

    # 3.2 Resolve com RCD
    try:
        x_rcd, f_rcd, it_rcd, t_rcd = solve_rcd(A, b, max_iteracoes=max_iter, tolerancia=tol)
        results['rcd'] = {'x': x_rcd, 'f': f_rcd, 'iter': it_rcd, 'time': t_rcd}
        np.savez('lsq_results_rcd.npz', x_star=x_rcd, f_x_star=f_rcd, iterations=it_rcd, time_sec=t_rcd)
        print("Resultado RCD salvo em lsq_results_rcd.npz")
    except Exception as e:
        print(f"Erro ao executar RCD: {e}")

    # 3.3 Resolve com MDCD
    try:
        x_mdcd, f_mdcd, it_mdcd, t_mdcd = solve_mdcd(A, b, max_iteracoes=max_iter, tolerancia=tol)
        results['mdcd'] = {'x': x_mdcd, 'f': f_mdcd, 'iter': it_mdcd, 'time': t_mdcd}
        np.savez('lsq_results_mdcd.npz', x_star=x_mdcd, f_x_star=f_mdcd, iterations=it_mdcd, time_sec=t_mdcd)
        print("Resultado MDCD salvo em lsq_results_mdcd.npz")
    except Exception as e:
        print(f"Erro ao executar MDCD: {e}")

    # --- Resumo dos Resultados ---
    print("\n--- Resumo dos Resultados ---")
    for method, res in results.items():
        print(f"Método: {method.upper()}")
        print(f"  Iterações: {res['iter']}")
        print(f"  Tempo (s): {res['time']:.4f}")
        print(f"  f(x*): {res['f']:.6e}")
        # print(f"  ||x*||: {np.linalg.norm(res['x']):.4f}") # Opcional: norma da solução

# --- Ponto de Entrada ---
if __name__ == "__main__":
    # Define o nome do arquivo de entrada aqui
    # Deveria vir do upload ou argumento, mas fixamos para teste
    input_filename = "lsq_10000_1e-4_1.txt" # Substitua pelo nome real do arquivo .txt

    # Verifica se o arquivo existe antes de chamar main
    MAX_ITERATIONS = 20000 # Aumentado para dar mais chance de convergência
    TOLERANCE = 1e-7

    if os.path.exists(input_filename):
         # Parâmetros de execução
         main(input_filename, max_iter=MAX_ITERATIONS, tol=TOLERANCE)
    else:
         print(f"Erro: Arquivo de entrada '{input_filename}' não encontrado.")
         print("Por favor, certifique-se de que o arquivo .txt com os dados LSQ")
         print("esteja no diretório  e atualize a variável 'input_filename'.")
         # Tentativa de usar o arquivo original do upload se o nome fixo falhar
         original_upload_name = "pasted_content.txt" # Nome do arquivo original do usuário
         if input_filename != original_upload_name and os.path.exists(original_upload_name):
             print(f"Tentando usar o arquivo original: {original_upload_name}")
             main(original_upload_name, max_iter=MAX_ITERATIONS, tol=TOLERANCE)
         else:
             print("Não foi possível encontrar um arquivo de entrada válido.")