# Large-Scale Optimization Solver

Este projeto implementa um solver de Programação Linear (LP) que lê arquivos no formato **MPS**, resolve o problema usando métodos de otimização e exibe os resultados em uma interface web (Streamlit).

---

## 📋 Pré-requisitos
- Python 3.8+
- Bibliotecas: `streamlit`, `numpy`, `scipy`, `regex`
- (Opcional) Ambiente virtual (`venv` ou `conda`)

---

## 🚀 Instalação

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/joaoh224488/Large-Scale-Optimization.git
   cd Large-Scale-Optimization
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Interface Gráfica (Streamlit)

Execute o solver via navegador:
```bash
streamlit run Interface.py
```
**Passos**:
1. Acesse a URL exibida no terminal (geralmente `http://localhost:8501`).
2. Selecione um arquivo `.mps` da pasta `Instancias/`.
3. Aguarde a solução do problema.
4. Veja os resultados na tela (status, valor objetivo, iterações, etc.).

---

## 🛠️ Ferramentas

### Biblioteca para Computação Científica
- [SciPy](https://docs.scipy.org/doc/scipy/)  
  Método `linprog`: [scipy.optimize.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)

### Solver para Otimização Linear de Larga Escala
- [HiGHS](https://pypi.org/project/highspy/)  
  Documentação: [HiGHS Python Interface](https://ergo-code.github.io/HiGHS/dev/interfaces/python/)

### Otimizador de Pontos Interiores
- [IPOPT](https://pypi.org/project/ipopt/)  
  Guia de uso: [PyOptSparse/IPOPT](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html)

### Modelagem de Problemas de Otimização
- [Pyomo](http://www.pyomo.org/)  

---

## Equipe

*  Integrantes da Equipe

---
