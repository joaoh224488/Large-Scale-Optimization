import os
import requests
import logging
from bs4 import BeautifulSoup

class get_data():

    def __init__(self):
        
        # URL da página a ser analisada
        self.url = 'https://netlib.org/lp/data/index.html'
        self.save_dir = "Instancias"


    def save_data(self):

        # Realiza a requisição HTTP para obter o conteúdo da página
        response = requests.get(self.url)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Analisa o conteúdo HTML da página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Encontra todas as tags <a> com o atributo href
        links = soup.find_all('a', href=True)

        # Itera sobre os links encontrados
        for link in links:
            href = link['href']
            # Verifica se o href é um link relativo
            if not href.startswith('http'):
                href = os.path.join(os.path.dirname(self.url), href)

            logging.info("Atual href:\t",href)
            # Obtém o nome do arquivo a partir do link
            filename = os.path.basename(href)



            # Realiza a requisição para baixar o conteúdo do link
            file_response = requests.get(href)
            file_response.raise_for_status()


            # Define o diretório de destino
            
            os.makedirs(self.save_dir, exist_ok=True)  # Cria o diretório se não existir

            # Garante que o arquivo tenha a extensão .mps
            if not filename.endswith(".mps"):
                
                filename = filename.split(".")[0]
                filename += ".mps"

            # Salva o conteúdo no arquivo correspondente dentro de /instancias
            file_path = os.path.join(self.save_dir, filename)
            with open(file_path, 'wb') as file:
                file.write(file_response.content)
            print(f'Arquivo {file_path} salvo com sucesso.')


if __name__ == "__main__":

    logging.info("INIT SAVE DATA")
    d = get_data()
    d.save_data()
    logging.info("END SAVE DATA")