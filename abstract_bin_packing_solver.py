
from abc import ABC, abstractmethod

import torch


class AbstractBinPackingSolver(ABC):
    def __init__(
            self,
            train_loader,
            val_loader,
            log_fn,
            device
        ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_fn = log_fn
        self.device = device
        
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def inference(self):
        pass