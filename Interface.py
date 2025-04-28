import streamlit as st
from codes.read_instance_regex import MPSParser
from codes.Solvers.Linprog_solver import LinprogSolver
from codes.Solvers.HighsSolver import HighsSolver
import tempfile
import os
import time

def main():
    st.set_page_config(page_title="Solver de PL", layout="centered")
    st.title("Solver de Programação Linear")
    
    if "file_path" not in st.session_state:
        st.session_state.file_path = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    uploaded_file = st.file_uploader("Escolha um arquivo MPS", type=["mps"])
    
    if uploaded_file is not None:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mps") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            st.session_state.file_path = tmp_file.name
        
        if st.button("Confirmar e Resolver"):
            st.session_state.page = "results"
            st.session_state.processing = True
            st.rerun()


def results_page():
    st.set_page_config(page_title="Resultados", layout="centered")
    st.title("Resultados da Otimização")
    
    if "file_path" not in st.session_state or st.session_state.file_path is None:
        st.error("Nenhum arquivo foi enviado.")
        
        if st.button("Voltar para o início"):
            st.session_state.page = "main"
            st.rerun()
        
        return
    
    if st.session_state.processing:
        st.write("Processando a otimização...")
        progress_bar = st.progress(0)
        
        try:
            solver = HighsSolver(st.session_state.file_path)
            
            for i in range(1, 101):
                time.sleep(0.02)  # Simula progresso
                progress_bar.progress(i)
            
            solver.run()
            results = solver.get_results()
            
            st.session_state.results = results
            st.session_state.processing = False
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar o arquivo MPS: {e}")
            st.session_state.processing = False
            st.rerun()
    
    else:
        results = st.session_state.get("results", None)
        
        if results:
            st.write(f"**Status:** {results['status']}")
            st.write(f"**Valor objetivo:** {results['objective_value']}")
            st.write(f"**Sucesso:** {results['success']}")
            st.write(f"**Número de iterações:** {results['iterations']}")
        
        else:
            st.error("Erro ao resolver o problema.")
    
    if st.button("Voltar para o início"):
        
        if os.path.exists(st.session_state.file_path):
            os.unlink(st.session_state.file_path)
        
        st.session_state.file_path = None
        st.session_state.results = None
        st.session_state.page = "main"
        st.rerun()

# Gerenciar navegação entre páginas
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main()
else:
    results_page()