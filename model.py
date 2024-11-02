from typing import List
from ortools.sat.python import cp_model
import hashlib

class BaseBinPackingGameResult:
    pass
    
class ResolvedBinPackingGameResult(BaseBinPackingGameResult):
    def __init__(self, positions: List[tuple[int, int]]):
        self.positions = positions
    
    def __str__(self) -> str:
        return f"ResolvedBinPackingGameResult(positions={self.positions})"
    
    def __repr__(self) -> str:
        return self.__str__()

class InfeasibleBinPackingGameResult(BaseBinPackingGameResult):
    def __str__(self) -> str:
        return "UnfeasibleBinPackingGameResult()"
    
    def __repr__(self) -> str:
        return self.__str__()

class InvalidBinPackingGameResult(BaseBinPackingGameResult):
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

class BinPackingGame:
    def __init__(self, container: Box, boxes: List[Box]):
        self.container = container
        self.boxes = boxes
    
    def __str__(self) -> str:
        return f"BinPackingGame(container={self.container}, boxes={self.boxes})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def generate_unique_key(self) -> str:
        # Convert container and boxes attributes to a unique string
        container_str = f"{self.container.width},{self.container.height}"
        boxes_str = ";".join(f"{box.width},{box.height}" for box in self.boxes)
        
        # Combine all data into a single string and hash it
        unique_data = f"{container_str}|{boxes_str}"
        return hashlib.md5(unique_data.encode()).hexdigest()
    
    def coverage(self) -> float:
        return sum(box.width * box.height for box in self.boxes) / (self.container.width * self.container.height)

    def solve(self) -> BaseBinPackingGameResult:
        model = cp_model.CpModel()
        # Crear variables para las coordenadas (x, y) de la esquina inferior izquierda de cada caja
        x = [model.NewIntVar(0, self.container.width - w, f'x_{i}') for i, (w, h) in enumerate(self.boxes)]
        y = [model.NewIntVar(0, self.container.height - h, f'y_{i}') for i, (w, h) in enumerate(self.boxes)]

        # Añadir restricciones para evitar superposición entre cajas
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                w_i, h_i = self.boxes[i]
                w_j, h_j = self.boxes[j]
                
                # Crear variables booleanas para representar cada posible no-superposición
                no_overlap_right = model.NewBoolVar(f'no_overlap_right_{i}_{j}')
                no_overlap_left = model.NewBoolVar(f'no_overlap_left_{i}_{j}')
                no_overlap_above = model.NewBoolVar(f'no_overlap_above_{i}_{j}')
                no_overlap_below = model.NewBoolVar(f'no_overlap_below_{i}_{j}')
                
                # Añadir restricciones que conecten las posiciones de las cajas con las variables booleanas
                model.Add(x[i] + w_i <= x[j]).OnlyEnforceIf(no_overlap_right)
                model.Add(x[j] + w_j <= x[i]).OnlyEnforceIf(no_overlap_left)
                model.Add(y[i] + h_i <= y[j]).OnlyEnforceIf(no_overlap_above)
                model.Add(y[j] + h_j <= y[i]).OnlyEnforceIf(no_overlap_below)
                
                # Asegurarse de que al menos una de las condiciones de no-superposición es verdadera
                model.AddBoolOr([no_overlap_right, no_overlap_left, no_overlap_above, no_overlap_below])

        # Resolver el modelo
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Mostrar el resultado
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            return ResolvedBinPackingGameResult([(solver.Value(x[i]), solver.Value(y[i])) for i in range(len(self.boxes))])
        elif status == cp_model.INFEASIBLE:
            return InfeasibleBinPackingGameResult()
        else:
            return InvalidBinPackingGameResult()