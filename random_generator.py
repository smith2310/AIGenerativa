import random
from typing import List
from model import BinPackingGame, Box, ResolvedBinPackingGameResult
import copy


class RandomGenerator:
    def __init__(
            self,
            container: Box,
            container_coverage: float = 0.5,
            min_boxes_per_container: int = 1
        ):
        self.container = container
        self.container_coverage = container_coverage
        self.max_boxes_per_container = self.container.width * self.container.height
        self.min_boxes_per_container = min_boxes_per_container
        assert self.min_boxes_per_container <= self.max_boxes_per_container, "min_boxes_per_container must be less than or equal to the container's capacity"

    def generate(self, games_to_generate: int) -> List[BinPackingGame]:
        """
        Vamos a generar una lista con la cantidad de cajas por contenedor,
        desde el min 1 caja hasta el max de cajas que caben en el contenedor (ancho * alto) cajas de 1x1.
        De esta lista aleatoriamente vamos ir generando cajas con ancho y alto aleatorio, poniendolas en el contenedor
        y chequeando si es una solucion valida y sino la tenemos ya en la lista de soluciones válidas,
        de ser asi la agregamos a la lista.
        """
        valid_bin_packing_games = []
        generated_bin_packing_game_keys = {}
        boxes_per_container = list(range(self.min_boxes_per_container, self.max_boxes_per_container + 1))
        while (len(valid_bin_packing_games) <= games_to_generate):
            boxes = []
            number_of_boxes = random.choice(boxes_per_container)
            # print(f'Attempting to create bin packing game with {number_of_boxes} boxes')
            for _ in range(number_of_boxes):
                width = random.randint(1, self.container.width)
                height = random.randint(1, self.container.height)
                boxes.append(Box(width, height))
            bin_packing_game = BinPackingGame(self.container, boxes)
            key = bin_packing_game.generate_unique_key()
            if (key not in generated_bin_packing_game_keys) and (bin_packing_game.coverage() >= self.container_coverage):
                result = bin_packing_game.solve()
                if isinstance(result, ResolvedBinPackingGameResult):
                    valid_bin_packing_games.append(bin_packing_game)
                    generated_bin_packing_game_keys[key] = True
                    print(f'Bin packing game with {number_of_boxes} boxes generated successfully!')
        return valid_bin_packing_games
        