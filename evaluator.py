from typing import List
from bin_packing_dataset import BinPackingDataset
from models import BinPackingGame, Box, ResolvedBinPackingGameResult
import random


class EvalutionResult:
    def __init__(
            self,
            container: Box,
            valid_games_percentage,
            new_games_percentage,
            unique_games_percentage,
            coverage_average,
            box_count_average
        ):
        """
        Args:
            container (Box): Caja contenedora de los juegos generados.
            valid_games_average (float): Promedio de juegos válidos.
            new_games_average (float): Promedio de juegos nuevos (no conocidos durante el entrenamiento).
            unique_games_average (float): Promedio de juegos únicos generados para la configuracion (container)
            coverage_average (float): Promedio de cobertura.
            box_count_average (float): Promedio de cantidad de cajas para el contenedor
        """
        self.container = container
        self.valid_games_percentage = valid_games_percentage
        self.new_games_percentage = new_games_percentage
        self.unique_games_percentage = unique_games_percentage
        self.coverage_average = coverage_average
        self.box_count_average = box_count_average
    
    def __str__(self):
        return f"Container: {self.container}, Valid Games Average: {self.valid_games_average}, Unique Games Average: {self.unique_games_average}, Coverage Average: {self.coverage_average}, Box Count Average: {self.box_count_average}, New Games Average: {self.new_games_average}"
    

def tensor_to_box(tensor):
    return Box(int(tensor[0].item()), int(tensor[1].item()))

class Evaluator:
    def __init__(
            self,
            sequence_generator,
            dataset: BinPackingDataset,
            max_sequence_length,
            max_dim,
            configs_to_evaluate,
            attempt_per_config = 10,
            min_coverage = None
    ):
        """
        Args:
            sequence_generator (function): Función que recibe un Box (container) y un largo máximo de sequencia y retorna una lista de Box
            dataset (BinPackingDataset): Dataset contra el cual se evalua si un juego generado es "nuevo" o ya es conocido por el dataset.
            max_sequence_length (int): Longitud máxima de las secuencias generadas.
            max_dim (int): Dimensión máxima de las cajas generadas en caja eje de las lista generadas.
            configs_to_evaluate (int): cantidad de configuraciones a evaluar.
            attempt_per_config (int): Número de intentos por configuración.
            min_coverage (float|None): Cobertura mínima que se debe alcanzar.
        """
        self.sequence_generator = sequence_generator
        self.dataset = dataset
        self.max_sequence_length = max_sequence_length
        self.max_dim = max_dim
        assert configs_to_evaluate <= max_dim * max_dim, "La cantidad de configuraciones a evaluar debe ser menor o igual a la cantidad de configuraciones posibles."
        self.configs_to_evaluate = configs_to_evaluate
        self.attempt_per_config = attempt_per_config
        self.min_coverage = min_coverage

    def evaluate(self) -> List[EvalutionResult]:
        """
        Evalua la generación de juego dado por configs_to_evaluate y attempt_per_config.
        """
        possible_containers = [Box(i, j) for i in range(1, self.max_dim + 1) for j in range(1, self.max_dim + 1)]
        evalution_results = []
        for _ in range(self.configs_to_evaluate):
            # Obtenemos una posible configuración y luego la quitamos de la lista de posibles configuraciones.
            container = random.choice(possible_containers)
            possible_containers.remove(container)
            valid_games_average = 0
            new_games_average = 0
            coverage_average = 0
            box_count_average = 0
            unique_games = set()
            for _ in range(self.attempt_per_config):
                sequence_lenght = random.randint(1, self.max_sequence_length + 1)
                boxes = self.sequence_generator(container, sequence_lenght)
                valid_boxes = [box for box in boxes if box.width > 0 and box.height > 0]
                if len(valid_boxes) == 0:
                    continue
                game = BinPackingGame(container, valid_boxes)
                result = game.solve()
                game_coverage = game.coverage()
                if isinstance(result, ResolvedBinPackingGameResult):
                    valid_games_average += 1
                    if not self.dataset.has_game(game):
                        unique_games.add(game)
                    if not self.dataset.has_game(game):
                        new_games_average += 1
                    if self.min_coverage is None or game_coverage >= self.min_coverage:
                        coverage_average += game_coverage
                    box_count_average += len(valid_boxes)
            if valid_games_average != 0:    
                coverage_average /= valid_games_average
                box_count_average /= valid_games_average
            valid_games_average /= self.attempt_per_config
            unique_games_average = len(unique_games)/self.attempt_per_config
            new_games_average /= self.attempt_per_config
            evalution_results.append(
                EvalutionResult(
                    container,
                    valid_games_percentage = valid_games_average,
                    new_games_percentage = new_games_average, 
                    unique_games_percentage = unique_games_average,
                    coverage_average = coverage_average,
                    box_count_average = box_count_average
                )
            )
        return evalution_results
        
                    