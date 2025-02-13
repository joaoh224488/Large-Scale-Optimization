from mip import Model


if __name__ == "__main__":

    # Carregar o modelo MPS
    model = Model()
    model.read("Instancias/israel.mps")

    # Exibir detalhes do modelo
    print(model)
