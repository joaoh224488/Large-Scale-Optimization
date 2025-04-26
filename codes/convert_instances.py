import os
from highspy import Highs

class MpsToLpConverter:
    def __init__(self, mps_path=None, instances_folder=None):
        """
        Inicializa o conversor com os caminhos necessários.
        
        Args:
            mps_path (str, optional): Caminho completo para o arquivo .mps. Defaults to None.
            instances_folder (str, optional): Caminho para a pasta de instâncias. Defaults to None.
        """
        self.mps_path = mps_path
        self.instances_folder = instances_folder
        self.lp_path = None
        
        if self.mps_path is not None:
            self._set_lp_path()
    
    def set_mps_path(self, mps_path):
        """Define o caminho do arquivo .mps e atualiza o caminho do .lp correspondente."""
        self.mps_path = mps_path
        self._set_lp_path()
    
    def set_instances_folder(self, instances_folder):
        """Define o caminho da pasta de instâncias."""
        self.instances_folder = instances_folder
        if self.mps_path is not None:
            self._set_lp_path()
    
    def _set_lp_path(self):
        """Define o caminho do arquivo .lp baseado no caminho do .mps e da pasta de instâncias."""
        if self.instances_folder is None:
            folder = os.path.dirname(self.mps_path)
        else:
            folder = self.instances_folder
            
        filename = os.path.splitext(os.path.basename(self.mps_path))[0] + '.lp'
        self.lp_path = os.path.join(folder, filename).replace("\\", "/")
    
    def convert(self):
        """
        Converte o arquivo .mps para .lp usando o HiGHS.
        
        Returns:
            str: Caminho do arquivo .lp gerado
        """
        if self.mps_path is None:
            raise ValueError("Caminho do arquivo .mps não foi definido")
        
        if not os.path.exists(self.mps_path):
            raise FileNotFoundError(f"Arquivo .mps não encontrado: {self.mps_path}")
        
        # Criar pasta de saída se não existir
        if self.instances_folder is not None:
            os.makedirs(self.instances_folder, exist_ok=True)
        
        highs = Highs()
        highs.readModel(self.mps_path)
        highs.writeModel(self.lp_path)
        
        return self.lp_path
    
    def get_lp_path(self):
        """Retorna o caminho do arquivo .lp que será/foi gerado."""
        return self.lp_path
    
    def get_mps_path(self):
        """Retorna o caminho do arquivo .mps."""
        return self.mps_path
    
    def get_instances_folder(self):
        """Retorna o caminho da pasta de instâncias."""
        return self.instances_folder


# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo 1: Especificando apenas o caminho do MPS
    converter1 = MpsToLpConverter(mps_path='/home/vaio/ufpb/Large-Scale-Optimization/Instancias/mps/25fv47.mps',instances_folder="/home/vaio/ufpb/Large-Scale-Optimization/Instancias/lp")
    lp_path1 = converter1.convert()
    print(f"Arquivo LP gerado em: {lp_path1}")
    
  
