import os
import requests
from bs4 import BeautifulSoup

class MPSDownloader:
    def __init__(self, url, save_dir):
        self.url = url
        self.save_dir = save_dir
        self.base_url = 'https://github.com'
        self.url_raw = "https://raw.githubusercontent.com/ozy4dm/lp-data-netlib/main/mps_files/"
        self.mps_links = []

    def fetch_mps_links(self):
        # Realiza a requisição HTTP para obter o conteúdo da página
        response = requests.get(self.url)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Analisa o conteúdo HTML da página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Encontra todas as tags <a> com o atributo href que contêm links para arquivos .mps
        links = soup.find_all('a', href=True)
        self.mps_links = [self.base_url + link['href'] for link in links if link['href'].endswith('.mps')]

    def download_files(self):
        # Obtém o nome do arquivo a partir do link
        filenames = [os.path.basename(link) for link in self.mps_links]

        for filename in filenames:
            response = requests.get(self.url_raw + filename)

            if response.status_code == 200:
                # Salvar o conteúdo do arquivo
                with open(os.path.join(self.save_dir, filename), 'wb') as file:
                    file.write(response.content)
                print(f"Arquivo {filename} baixado com sucesso.")
            else:
                print(f"Falha ao baixar {filename}: {response.status_code}")

    def run(self):
        self.fetch_mps_links()
        self.download_files()


# Uso da classe
if __name__ == "__main__":
    url = 'https://github.com/ozy4dm/lp-data-netlib/tree/main/mps_files'
    save_dir = "Instancias"
    
    downloader = MPSDownloader(url, save_dir)
    downloader.run()