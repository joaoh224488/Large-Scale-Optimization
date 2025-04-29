import os
from datetime import datetime

def generate_output_file(output_folder, problem_name, solver_results, primal_vars=None, dual_vars=None):
    """
    Gera um arquivo de texto formatado com os resultados da otimização.
    
    Args:
        output_folder (str): Pasta onde o arquivo será salvo.
        problem_name (str): Nome do problema (ex: "Blend.mp").
        solver_results (dict): Resultados gerais (ex: valor ótimo, iterações, viabilidade, tempo).
        primal_vars (list of tuples): Lista [(índice, valor primal, preço dual)].
        dual_vars (list of tuples): Lista [(índice, folga, valor dual)].
    """

    # Criar pasta se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Nome do arquivo baseado no problema e timestamp
    safe_problem_name = os.path.splitext(problem_name)[0]
    file_path = os.path.join(output_folder, f"{safe_problem_name}.mps")

    with open(file_path, 'w') as f:
        f.write(f"{problem_name}\n\n\n")
        f.write(" " * 20 + "RESULTADO FINAL\n\n")
        f.write("_" * 48 + "\n")
        f.write("-" * 50 + "\n")

        f.write(f"    VALOR OTIMO PRIMAL = {solver_results.get('valor_otimo_primal', 0):>20.10E}\n")
        f.write(f"         ITERATIONS    = {solver_results.get('iterations', 0)}\n")
        f.write(f"         GAP           = {solver_results.get('gap', 0):>20.10E}\n")
        f.write(f"    VALOR OTIMO DUAL   = {solver_results.get('valor_otimo_dual', 0):>20.10E}\n")
        f.write(f"    VIABILIDADE PRIMAL = {solver_results.get('viabilidade_primal', 0):>20.10E}\n")
        f.write(f"    VIABILIDADE DUAL   = {solver_results.get('viabilidade_dual', 0):>20.10E}\n")
        f.write(f"         TEMPO (SEG.)  = {solver_results.get('tempo', 0):>20.10E}\n")

        f.write("_" * 48 + "\n\n")
        f.write("_" * 65 + "\n")
        f.write("-" * 67 + "\n")
        f.write("     VAR.           SOL. PRIMAL                 DUAL_PRICES\n")
        f.write("_" * 65 + "\n")

        if primal_vars:
            for idx, primal_value, dual_price in primal_vars:
                f.write(f"{idx:>8} {primal_value:>25.10f} {dual_price:>30.5f}\n")
        else:
            f.write("Nenhuma variável encontrada.\n")

        f.write("_" * 65 + "\n\n")
        f.write("_" * 65 + "\n")
        f.write("-" * 67 + "\n")
        f.write("    RESTR            FOLGAS                    SOL. DUAL\n")
        f.write("_" * 65 + "\n")

        if dual_vars:
            for idx, folga, sol_dual in dual_vars:
                f.write(f"{idx:>8} {folga:>25.10f} {sol_dual:>30.5f}\n")
        else:
            f.write("Nenhuma restrição encontrada.\n")

        f.write("_" * 65 + "\n")

    return file_path
