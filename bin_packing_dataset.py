import os
import torch
from torch.utils.data import Dataset

from models import BinPackingGame

class BinPackingDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Ruta al directorio que contiene los archivos .txt.
        """
        self.data = []
        self._keys = set()

        # Recorrer todos los archivos en el directorio
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Asegurarse de procesar solo archivos .txt
            if os.path.isfile(file_path) and file_path.endswith('.txt'):
                with open(file_path, 'r') as file:
                    for line in file:
                        values = list(map(float, line.strip().split(',')))  # Convertir a float
                        container = values[:2]  # El primer par es el contenedor (width, height)
                        boxes = [(values[i], values[i+1]) for i in range(2, len(values), 2)]  # Pares de cajas
                        self.data.append((container, boxes))
                        key = hash((tuple(container), tuple(boxes)))
                        self._keys.add(key)

    def has_game(self, game: BinPackingGame) -> bool:
        """
        Verifica si el juego ya está en el dataset.
        
        Args:
            game (BinPackingGame): Juego a verificar.
            
        Returns:
            bool: True si el juego ya está en el dataset, False en caso contrario.
        """
        container = (float(game.container.width), float(game.container.height))
        boxes = [(float(box.width), float(box.height)) for box in game.boxes]
        key = hash((tuple(container), tuple(boxes)))
        return key in self._keys

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Índice del elemento a obtener.
            
        Returns:
            tuple: (contenedor, cajas) donde:
                - contenedor: Tensor de tamaño (2,).
                - cajas: Tensor de tamaño (n, 2), donde n es el número de cajas.
        """
        container, boxes = self.data[idx]
        container_tensor = torch.tensor(container, dtype=torch.float32)
        boxes_with_eos = boxes + [(0, 0)]
        boxes_tensor = torch.tensor(boxes_with_eos, dtype=torch.float32)
        return container_tensor, boxes_tensor