import streamlit as st
from read_instance_regex import MPSParser
from run_linprog import linprog_solver
import logging

def main():
    st.title("Solver de Programação Linear")
    
    uploaded_file = st.file_uploader("Escolha um arquivo MPS", type=["mps"])
    
    if uploaded_file is not None:
        # Salvar o arquivo temporariamente
        instance_path = f"/tmp/{uploaded_file.name}"
        with open(instance_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Criar uma instância do solver
            solver = linprog_solver(instance_path)
            
            # Executar o solver
            solver.run()
            
            # Obter e exibir os resultados
            results = solver.get_results()
            
            if results:
                st.subheader("Resultados da Otimização")
                st.write(f"Status: {results['status']}")
                st.write(f"Valor objetivo: {results['objective_value']}")
                st.write(f"Sucesso: {results['success']}")
                st.write(f"Número de iterações: {results['iterations']}")
            else:
                st.error("Erro ao resolver o problema.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo MPS: {e}")

if __name__ == "__main__":
    main()