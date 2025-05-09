import os
import time
import tempfile
import streamlit as st

from typing import Optional, Dict, Any
from codes.Solvers.Highs import HighsSolver
from codes.Solvers.MirrorDescentSolver import MirrorDescentSolver  # importe seu novo solver aqui


def cleanup_temp_file(file_path: Optional[str]) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            st.error(f"Erro ao remover arquivo temporário: {e}")


def main_page() -> None:
    st.set_page_config(page_title="Solver de PL", layout="centered")
    st.title("Solver de Problemas de Otimização")

    uploaded_file = st.file_uploader("Escolha um arquivo", type=["mps", "txt"])

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
    if "MODEL NAME" in results:
        # Exibir resultados do HighsSolver
        st.write(f"**MODEL NAME:** {results['MODEL NAME']}")
        st.write(f"**STATUS:** {results['STATUS']}")
        st.write(f"**VALOR ÓTIMO PRIMAL:** {results['VALOR ÓTIMO PRIMAL']}")
        st.write(f"**VALOR ÓTIMO DUAL:** {results['VALOR ÓTIMO DUAL']}")
        st.write(f"**GAP ABSOLUTO:** {results['GAP ABSOLUTO']}")
        st.write(f"**GAP RELATIVO:** {results['GAP RELATIVO']}")
        st.write(f"**INVIABILIDADE PRIMAL:** {results['INVIABILIDADE PRIMAL']}")
        st.write(f"**INVIABILIDADE DUAL:** {results['INVIABILIDADE DUAL']}")
        st.write(f"**ITERAÇÕES:** {results['ITERAÇÕES']}")
        st.write(f"**TEMPO(SEG.):** {results['TEMPO(SEG.)']}")
        st.table(results['Df1'])
        st.table(results['Df2'])
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
            solver_type = st.selectbox(
                "Selecione o solver para otimização",
                ["HighsSolver", "MirrorDescentSolver"],
                key="solver_selection"
            )



            # Configurações específicas para MirrorDescentSolver
            if solver_type == "MirrorDescentSolver":
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
                if solver_type == "HighsSolver":
                    solver = HighsSolver(file_path)
                elif solver_type == "MirrorDescentSolver":
                    solver = MirrorDescentSolver(
                        file_path,
                        max_iter=max_iter,
                        versao=versao
                    )
                else:
                    raise ValueError("Solver não suportado.")

                # Simulação de progresso
                for i in range(1, 101):
                    time.sleep(0.01)
                    progress_bar.progress(i)

                solver.run()
                results = solver.get_results()

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
