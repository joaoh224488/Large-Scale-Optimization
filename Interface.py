import os
import time
import tempfile
import streamlit as st

from typing import Optional, Dict, Any
from codes.Solvers.HighsSolver import HighsSolver

# Remove os arquivos temporários, se houverem
def cleanup_temp_file(file_path: Optional[str]) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            st.error(f"Erro ao remover arquivo temporário: {e}")


# Página inicial, onde o usuário pode fazer upload do arquivo MPS
def main_page() -> None:
    st.set_page_config(page_title="Solver de PL", layout="centered")
    st.title("Solver de Problemas de Otimização de Grande Porte")
    
    uploaded_file = st.file_uploader("Escolha um arquivo MPS", type=["mps"])
    
    if uploaded_file is not None:
        original_filename = uploaded_file.name
        
        # Verifica se o arquivo está vazio
        if uploaded_file.size == 0:
            st.error("O arquivo está vazio.")
            return
            
        try:
            # Cria o arquivo temporário no diretório temporário com o nome original
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, original_filename)
            
            # Salva o conteúdo do arquivo uploadado
            with open(temp_file_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
            
            st.session_state.file_path = temp_file_path
            
            if st.button("Confirmar e Resolver"):
                st.session_state.page = "results"
                st.session_state.processing = True
                st.rerun()
                
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            cleanup_temp_file(st.session_state.get("file_path"))


# Exibe os resultados de otimização na tela
def display_results(results: Dict[str, Any]) -> None:
    st.write(f"**MODEL NAME:** {results['MODEL NAME']}")
    st.write(f"**STATUS:** {results['STATUS']}")
    st.write(f"**VALOR ÓTIMO PRIMAL:** {results['VALOR ÓTIMO PRIMAL']}")
    st.write(f"**VALOR ÓTIMO DUAL:** {results['VALOR ÓTIMO DUAL']}")
    st.write(f"**GAP:** {results['GAP']}")
    st.write(f"**INVIABILIDADE PRIMAL:** {results['INVIABILIDADE PRIMAL']}")
    st.write(f"**INVIABILIDADE DUAL:** {results['INVIABILIDADE DUAL']}")
    st.write(f"**ITERAÇÕES:** {results['ITERAÇÕES']}")
    st.write(f"**TEMPO(SEG.):** {results['TEMPO(SEG.)']}")
    st.table(results['Df1'])
    st.table(results['Df2'])

# Página de resultados: recebe o arquivo MPS, processa a otimização e exibe os resultados
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
            
            solver = HighsSolver(file_path)
            
            # Fake barra de progresso para simular o processamento
            for i in range(1, 101):
                time.sleep(0.01)
                progress_bar.progress(i)
            
            solver.run()
            results = solver.get_results()
            
            st.session_state.results = results
            st.session_state.processing = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao processar o arquivo MPS: {str(e)}")
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

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "main"

# Page routing
if st.session_state.page == "main":
    main_page()
else:
    results_page()