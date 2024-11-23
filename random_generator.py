import random
from typing import List, Optional
from models import BinPackingGame, Box, ResolvedBinPackingGameResult
import copy


class RandomGenerator:

    def __init__(
            self,
            container: Box,
            min_container_coverage: float = 0.5,
            max_container_coverage: float = 1.0,
            min_boxes_per_container: int = 1,
            max_boxes_per_container: Optional[int] = None
        ):
        self.container = container
        assert min_container_coverage <= 1.0, "container_coverage must be less than or equal to 1.0"
        assert min_container_coverage > 0.0, "container_coverage must be greater than 0.0"
        self.min_container_coverage = min_container_coverage
        assert max_container_coverage <= 1.0, "container_coverage must be less than or equal to 1.0"
        assert max_container_coverage >= min_container_coverage, f"container_coverage must be greater or equal than min_container_coverage={min_container_coverage}"
        self.max_container_coverage = max_container_coverage
        self.max_boxes_per_container = max_boxes_per_container or (container.width * container.height)
        maximum_boxes = container.width * container.height
        assert self.max_boxes_per_container <= maximum_boxes, f"max_boxes_per_container must be less than or equal to the container's capacity -> {maximum_boxes}"
        self.min_boxes_per_container = min_boxes_per_container
        assert self.min_boxes_per_container >= 1, "min_boxes_per_container must be greater than or equal to 1"
        assert self.min_boxes_per_container <= self.max_boxes_per_container, f"min_boxes_per_container must be less than or equal to max_boxes_per_container -> {self.max_boxes_per_container}"
        self.valid_bin_packing_games = []
        self.generated_bin_packing_game_keys = set()

    def add_game(self, bin_packing_game: BinPackingGame) -> bool:
        key = hash(bin_packing_game)
        if (len(bin_packing_game.boxes) >= self.min_boxes_per_container) \
            and (len(bin_packing_game.boxes) <= self.max_boxes_per_container) \
            and (key not in self.generated_bin_packing_game_keys) \
            and (bin_packing_game.coverage() >= self.min_container_coverage) \
            and (bin_packing_game.coverage() <= self.max_container_coverage):
            result = bin_packing_game.solve()
            if isinstance(result, ResolvedBinPackingGameResult):
                self.valid_bin_packing_games.append(bin_packing_game)
                self.generated_bin_packing_game_keys.add(key)
                return True
        return False

    def generate(self, games_to_generate: int) -> List[BinPackingGame]:
        self.valid_bin_packing_games = []
        self.generated_bin_packing_game_keys = set()
        """
        Vamos a generar una lista con la cantidad de cajas por contenedor,
        desde el min 1 caja hasta el max de cajas que caben en el contenedor (ancho * alto) cajas de 1x1.
        De esta lista aleatoriamente vamos ir generando cajas con ancho y alto aleatorio, poniendolas en el contenedor
        y chequeando si es una solucion valida y sino la tenemos ya en la lista de soluciones válidas,
        de ser asi la agregamos a la lista.
        """
        boxes_per_container = list(range(self.min_boxes_per_container, self.max_boxes_per_container + 1))
        attempts = 0
        while (len(self.valid_bin_packing_games) < games_to_generate):
            attempts += 1
            # Vamos a intentar generar un juego que contenga {number_of_boxes} cajas
            number_of_boxes = random.choice(boxes_per_container)
            boxes = []
            bin_packing_game = BinPackingGame(self.container, boxes)
            for _ in range(number_of_boxes):
                #Como condiciones de poda obtenemos un ancho aleatorio entre 1 y min(ancho del contenedor, espacio disponible en el contenedor)
                max_width = min(self.container.width, bin_packing_game.available_space())
                max_width = max(max_width, 1)
                width = random.randint(1, max_width)
                #Con ese ancho y dado el espacio disponible en el contenedor, obtenemos un alto aleatorio
                max_height = min(self.container.height, bin_packing_game.available_space() // width)
                max_height = max(max_height, 1)
                height = random.randint(1, max_height)
                bin_packing_game.boxes.append(Box(width, height))
            # Vamos a intentar agregar el juego a la lista de juegos válidos, si no es posible, le vamos quitando cajas hasta que sea posible
            while (len(bin_packing_game.boxes) > self.min_boxes_per_container) and (not self.add_game(bin_packing_game)):
                bin_packing_game.boxes.pop()
        print(f"Generated {len(self.valid_bin_packing_games)} valid bin packing games in {attempts} attempts")
        return self.valid_bin_packing_games
        