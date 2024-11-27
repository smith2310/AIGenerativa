from typing import List
from ortools.sat.python import cp_model
import hashlib

class BinPackingGameResult:
    pass
    
class ResolvedBinPackingGameResult(BinPackingGameResult):
    def __init__(self, positions: List[tuple[int, int]]):
        self.positions = positions
    
    def __str__(self) -> str:
        return f"ResolvedBinPackingGameResult(positions={self.positions})"
    
    def __repr__(self) -> str:
        return self.__str__()

class InfeasibleBinPackingGameResult(BinPackingGameResult):
    def __str__(self) -> str:
        return "UnfeasibleBinPackingGameResult()"
    
    def __repr__(self) -> str:
        return self.__str__()

class InvalidBinPackingGameResult(BinPackingGameResult):
    def __str__(self) -> str:
        return "InvalidBinPackingGameResult()"
    
    def __repr__(self) -> str:
        return self.__str__()

class Box:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"Box(width={self.width}, height={self.height})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __iter__(self):
        # Esto convierte `Box` en un iterable, permitiendo desempaquetarlo
        return iter((self.width, self.height))
    
    def __eq__(self, other):
        return self.width == other.width and self.height == other.height
    
    def __hash__(self):
        return hash((self.width, self.height))
    
    def rotate(self):
        return Box(self.height, self.width)

class BinPackingGame:
    def __init__(self, container: Box, boxes: List[Box]):
        self.container = container
        self.boxes = boxes
    
    def __str__(self) -> str:
        return f"BinPackingGame(container={self.container}, boxes={self.boxes})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        return hash((self.container, tuple(self.boxes)))
    
    def __eq__(self, value) -> bool:
        return self.container == value.container and self.boxes == value.boxes
    
    def coverage(self) -> float:
        return sum(box.width * box.height for box in self.boxes) / (self.container.width * self.container.height)
    
    def available_space(self) -> int:
        return self.container.width * self.container.height - sum(box.width * box.height for box in self.boxes)

    def solve(self) -> BinPackingGameResult:
        model = cp_model.CpModel()
        
        # Variables de coordenadas (x, y)
        x = [model.NewIntVar(0, self.container.width, f'x_{i}') for i in range(len(self.boxes))]
        y = [model.NewIntVar(0, self.container.height, f'y_{i}') for i in range(len(self.boxes))]
        
        # Variables para dimensiones efectivas (considerando rotación)
        width_eff = [model.NewIntVar(0, self.container.width, f'width_eff_{i}') for i in range(len(self.boxes))]
        height_eff = [model.NewIntVar(0, self.container.height, f'height_eff_{i}') for i in range(len(self.boxes))]
        is_rotated = [model.NewBoolVar(f'is_rotated_{i}') for i in range(len(self.boxes))]
        
        # Configurar dimensiones efectivas según rotación
        for i, (w, h) in enumerate(self.boxes):
            model.Add(width_eff[i] == w).OnlyEnforceIf(is_rotated[i].Not())
            model.Add(width_eff[i] == h).OnlyEnforceIf(is_rotated[i])
            model.Add(height_eff[i] == h).OnlyEnforceIf(is_rotated[i].Not())
            model.Add(height_eff[i] == w).OnlyEnforceIf(is_rotated[i])
            
            # Asegurarse de que las cajas no excedan los límites del contenedor
            model.Add(x[i] + width_eff[i] <= self.container.width)
            model.Add(y[i] + height_eff[i] <= self.container.height)
        
        # Restricciones de no superposición entre cajas
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                no_overlap_right = model.NewBoolVar(f'no_overlap_right_{i}_{j}')
                no_overlap_left = model.NewBoolVar(f'no_overlap_left_{i}_{j}')
                no_overlap_above = model.NewBoolVar(f'no_overlap_above_{i}_{j}')
                no_overlap_below = model.NewBoolVar(f'no_overlap_below_{i}_{j}')
                
                model.Add(x[i] + width_eff[i] <= x[j]).OnlyEnforceIf(no_overlap_right)
                model.Add(x[j] + width_eff[j] <= x[i]).OnlyEnforceIf(no_overlap_left)
                model.Add(y[i] + height_eff[i] <= y[j]).OnlyEnforceIf(no_overlap_above)
                model.Add(y[j] + height_eff[j] <= y[i]).OnlyEnforceIf(no_overlap_below)
                
                model.AddBoolOr([no_overlap_right, no_overlap_left, no_overlap_above, no_overlap_below])
        
        # Resolver el modelo
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Retornar las coordenadas finales si el modelo es factible
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            return ResolvedBinPackingGameResult([(solver.Value(x[i]), solver.Value(y[i])) for i in range(len(self.boxes))])
        elif status == cp_model.INFEASIBLE:
            return InfeasibleBinPackingGameResult()
        else:
            return InvalidBinPackingGameResult()
