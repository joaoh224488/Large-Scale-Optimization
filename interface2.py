import os
import time
import tempfile
import streamlit as st

from typing import Optional, Dict, Any
from codes.Solvers.Highs import HighsSolver
from codes.Solvers.MirrorDescentSolver import MirrorDescentSolver  # importe seu novo solver aqui
from codes.Solvers.DescendingByCoordinate import CoordinateDescent
from codes.Solvers.LocalOptimization import LocalOptimization
from codes.Solvers.GlobalOptimization import GlobalOptimization
from codes.instance_reader import read_instance
from codes.get_info_name import extract_size_tolerance


def cleanup_temp_file(file_path: Optional[str]) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            st.error(f"Erro ao remover arquivo temporário: {e}")



def main_page() -> None:
    st.set_page_config(page_title="Solver de PL", layout="centered")
    st.title("Solver de Problemas de Otimização")

    st.session_state.function_type = st.selectbox(
        "Selecione o tipo de função",
        ["Linear", "Quadrática"]
    )

    type_arq = ''
    methods = []
    if st.session_state.function_type == "Linear":
        type_arq = 'mps'
        methods = ["HiGHS"]
    else:
        type_arq = 'txt'
        methods = ["Gradiente Espelhado","Descida por Coordenada","Otimização Local","Otimização Global"]
    
    st.session_state.method_selected = st.selectbox(
        "Selecione o método de otimização",
        methods
    )

    



    uploaded_file = st.file_uploader("Escolha um arquivo", type=type_arq)

    if uploaded_file is not None:
        original_filename = uploaded_file.name

        if uploaded_file.size == 0:
            st.error("O arquivo está vazio.")
            return

        try:
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, original_filename)

            with open(temp_file_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())

            st.session_state.file_path = temp_file_path
            st.session_state.file_ext = os.path.splitext(original_filename)[1]

            if st.button("Confirmar e Resolver"):
                st.session_state.page = "results"
                st.session_state.processing = True
                st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            cleanup_temp_file(st.session_state.get("file_path"))


def display_results(results: Dict[str, Any]) -> None:

    print("results\n\n:",results)
    if "MODEL NAME" in results:
        # Exibir resultados do HighsSolver
        st.write(f"**MODEL NAME:** {results['MODEL NAME']}")
        st.write(f"**STATUS:** {results['STATUS']}")
        st.write(f"**VALOR ÓTIMO PRIMAL:** {results['VALOR ÓTIMO PRIMAL']}")
        st.write(f"**VALOR ÓTIMO DUAL:** {results['VALOR ÓTIMO DUAL']}")
        st.write(f"**GAP:** {results['GAP ABSOLUTO']}")
        st.write(f"**INVIABILIDADE PRIMAL:** {results['INVIABILIDADE PRIMAL']}")
        st.write(f"**INVIABILIDADE DUAL:** {results['INVIABILIDADE DUAL']}")
        st.write(f"**ITERAÇÕES:** {results['ITERAÇÕES']}")
        st.write(f"**TEMPO(SEG.):** {results['TEMPO(SEG.)']}")
        st.table(results['Df1'])
        st.table(results['Df2'])

   

    elif "message" in results:  # Resultados do LocalOptimization
        st.write("### Resultados da Otimização")
       

        st.write(f"**Status:** {results['message']}")
        st.write(f"**Valor da Função Objetivo:** {results['fun']:.6f}")
        # st.write(f"**Sucesso:** {'Sim' if results['success'] else 'Não'}")
        
        if 'nit' in results:
            st.write(f"**Número de Iterações:** {results['nit']}")
        if 'nfev' in results:
            st.write(f"**Avaliações da Função:** {results['nfev']}")
        if 'njev' in results:
            st.write(f"**Avaliações do Gradiente:** {results['njev']}")
        
        # Mostrar as primeiras 10 variáveis da solução para não poluir a tela
        st.write("**Solução (primeiras 10 variáveis):**")
        solution = results['x']
        st.write(solution[:10] if len(solution) > 10 else solution)

    elif "MELHOR_METODO" in results:  # Resultados do CoordinateDescent
        st.write("### Resultados da Descida por Coordenadas")
        
        # Mostra o melhor método
        st.write(f"**Melhor método:** {results['MELHOR_METODO']}")
        
        # Mostra a tabela comparativa
        st.write("**Comparação entre métodos:**")
        st.table(results['COMPARACAO'])
        
        # Mostra detalhes para cada método
        for method in ['CCD', 'RCD', 'MDCD']:
            if method in results:
                st.write(f"#### Método {method}")
                st.write(f"**Valor objetivo:** {results[method]['VALOR_OBJETIVO']:.6f}")
                st.write(f"**Iterações:** {results[method]['ITERACOES']}")
                st.write(f"**Tempo (s):** {results[method]['TEMPO (s)']:.4f}")
                st.write(f"**Norma do gradiente:** {results[method]['NORM_GRADIENTE']:.4e}")
                st.write(f"**Norma da solução:** {results[method]['NORM_SOLUCAO']:.4f}")
                
                # Mostra as primeiras 10 variáveis da solução
                st.write("**Solução (primeiras 10 variáveis):**")
                solution = results[method]['SOLUCAO_X']
                st.write(solution[:10] if len(solution) > 10 else solution)
    

    


    else:
        # Exibir resultados do MirrorDescentSolver
        st.write(f"**NORMA DO GRADIENTE:** {results['NORMA DO GRADIENTE']}")
        st.write(f"**CUSTO OBJETIVO CALCULADO:** {results['CUSTO OBJETIVO CALCULADO']}")
        st.write(f"**CUSTO OBJETIVO ESPERADO:** {results['CUSTO OBJETIVO ESPERADO']}")
        st.write(f"**GAP COM RELAÇÃO AO OTIMO:** {results['GAP COM RELAÇÃO AO OTIMO']}")
        st.write(f"**TEMPO (s):** {results['TEMPO (s)']}")
        st.write("**Comparação entre solução encontrada e solução ótima:**")
        st.table(results['Df'])


def results_page() -> None:
    st.set_page_config(page_title="Resultados", layout="centered")
    st.title("Resultados da Otimização")

    file_path = st.session_state.get("file_path")
    
    if not file_path:
        st.error("Nenhum arquivo foi enviado.")
        if st.button("Voltar para o início"):
            st.session_state.page = "main"
            st.rerun()
        return

    if st.session_state.get("processing", False):
        try:
            st.write("Processando a otimização...")
            progress_bar = st.progress(0)

            # Seleção do solver independentemente da extensão do arquivo
            # solver_type = st.selectbox(
            #     "Selecione o solver para otimização",
            #     ["HighsSolver", "MirrorDescentSolver"],
            #     key="solver_selection"
            # )


            solver_type = st.session_state.method_selected
            # Configurações específicas para MirrorDescentSolver

            # ["Gradiente Espelhado","Descida por Coordenada","Otimização Local","Otimização Global"]
            if solver_type == "Gradiente Espelhado":
                versao = st.selectbox(
                    "Escolha a versão do Mirror Descent",
                    ["negativa_entropia", "norma_p"],
                    key="mirror_version"
                )
                max_iter = st.number_input(
                    "Número máximo de iterações",
                    min_value=100,
                    max_value=100000,
                    value=10000,
                    key="max_iter"
                )

         
              

            # Botão para confirmar e iniciar a otimização
            if st.button("Iniciar Otimização"):
                if solver_type == "HiGHS":
                    solver = HighsSolver(file_path)
                    solver.run()
                    results = solver.get_results()



                elif solver_type == "Gradiente Espelhado":
                    solver = MirrorDescentSolver(
                        file_path,
                        max_iter=max_iter,
                        versao=versao
                    )
                    solver.run()
                    results = solver.get_results()

                elif solver_type == "Otimização Local":
                    A, b, x_star, solution_cost = read_instance(file_path)
                    n_vars = A.shape[1]
                    bounds = [(-2, 2) for _ in range(n_vars)]
                    solver = LocalOptimization(A, b, bounds=bounds)
                    solver.run(method="lsq")
                    results = solver.get_results()
                    # Adicionar o custo esperado para consistência com outros solvers
                    results['CUSTO OBJETIVO ESPERADO'] = solution_cost
                    results['GAP COM RELAÇÃO AO OTIMO'] = abs(results['fun'] - solution_cost)

                elif solver_type == "Otimização Global":
                    A, b, x_star, solution_cost = read_instance(file_path)
                    n_vars = A.shape[1]
                    bounds = [(-2, 2) for _ in range(n_vars)]
                    solver = GlobalOptimization(A, b, bounds=bounds)
                    solver.run(method="lsq")
                    results = solver.get_results()
                    # Adicionar informações adicionais para consistência
                    results['CUSTO OBJETIVO ESPERADO'] = solution_cost
                    results['GAP COM RELAÇÃO AO OTIMO'] = abs(results['fun'] - solution_cost)

                

                elif solver_type == "Descida por Coordenada":
                    solver = CoordinateDescent(file_path)
                    if solver.run(max_iter=10000, tol=1e-6):  # Você pode ajustar esses parâmetros
                        results = solver.get_results()
                        
                        # Adicionar o custo esperado se disponível (opcional)
                        if 'CUSTO OBJETIVO ESPERADO' in st.session_state:
                            for method in ['CCD', 'RCD', 'MDCD']:
                                if method in results:
                                    results[method]['GAP'] = abs(results[method]['VALOR_OBJETIVO'] - st.session_state['CUSTO OBJETIVO ESPERADO'])


                else:
                    raise ValueError("Solver não suportado.")

                # Simulação de progresso
                for i in range(1, 101):
                    time.sleep(0.01)
                    progress_bar.progress(i)

                

                st.session_state.results = results
                st.session_state.processing = False
                st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")
            st.session_state.processing = False
            cleanup_temp_file(file_path)
            st.rerun()
    else:
        results = st.session_state.get("results")
        if results:
            display_results(results)
        else:
            print("descinhecido",results)
            st.error("Erro ao resolver o problema.")

    if st.button("Voltar para o início"):
        cleanup_temp_file(file_path)
        st.session_state.clear()
        st.session_state.page = "main"
        st.rerun()

# Inicializa o estado da sessão
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
else:
    results_page()