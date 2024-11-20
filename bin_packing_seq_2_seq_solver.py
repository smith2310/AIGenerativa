
import copy
from abstract_bin_packing_solver import AbstractBinPackingSolver
import torch
import torch.nn as nn

from utils import EarlyStopping


class _BinPackingSeq2SeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(_BinPackingSeq2SeqEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)
    
class _BinPackingSeq2SeqDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(_BinPackingSeq2SeqDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc_to_hidden = nn.Linear(output_dim, hidden_dim)  # Nueva proyección

    def forward(self, z, seq_len):
        outputs = []
        hidden = (z.unsqueeze(0), z.unsqueeze(0))  # Initial hidden and cell states
        input_step = torch.zeros((z.size(0), 1, z.size(1))).to(z.device)  # Start token
        for _ in range(seq_len):
            output, hidden = self.lstm(input_step, hidden)
            box_dim = self.fc(output)  # Proyectar al espacio de salida (dim=2)
            outputs.append(box_dim.squeeze(1))
            input_step = torch.round(self.fc_to_hidden(box_dim)).detach()  # Proyectar de vuelta a hidden_dim
        return torch.stack(outputs, dim=1)

class _BinPackingSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(_BinPackingSeq2Seq, self).__init__()
        self.encoder = _BinPackingSeq2SeqEncoder(input_dim, hidden_dim)
        self.decoder = _BinPackingSeq2SeqDecoder(hidden_dim, output_dim)

    def forward(self, x, seq_len):
        z = self.encoder(x)
        return self.decoder(z, seq_len)
        
class BinPackingSeq2SeqSolver(AbstractBinPackingSolver):
    def __init__(
            self,
            train_loader,
            val_loader,
            log_fn,
            device
        ):
        super().__init__(train_loader, val_loader, log_fn, device)

    def evaluate(self, model, criterion):
        """
        Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

        Args:
            model (torch.nn.Module): El modelo que se va a evaluar.
            criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.

        Returns:
            float: La pérdida promedio en el conjunto de datos de evaluación.

        """
        model.eval()  # ponemos el modelo en modo de evaluacion
        val_loss = 0  # acumulador de la perdida
        with torch.no_grad():  # deshabilitamos el calculo de gradientes
            for x, y in self.val_loader:  # iteramos sobre el dataloader
                x = x.to(self.device)  # movemos los datos al dispositivo
                y = y.to(self.device)  # movemos los datos al dispositivo
                seq_len = y.size(1)
                output = model(x, seq_len)  # forward pass
                val_loss += criterion(output, y).item()  # acumulamos la perdida
        return val_loss / len(self.val_loader)  # retornamos la perdida promedio

        
    def train(self):
        model = _BinPackingSeq2Seq(input_dim=2, hidden_dim=128, output_dim=2).to(self.device)
        model.train()
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        epochs = 50
        train_loss = 0
        epoch_train_errors = []
        epoch_val_errors = []
        best_val_loss = float('inf')
        best_model_weights = None
        early_stopping = EarlyStopping(patience=5)
        do_early_stopping = False
        log_every = 2
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, y in self.train_loader:  # x: (batch_size, 2), y: (batch_size, seq_len, 2)
                x = x.to(self.device)
                y = y.to(self.device)
                seq_len = y.size(1)
                optimizer.zero_grad()
                y_pred = model(x, seq_len)
                batch_loss = criterion(y_pred, y)
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
            train_loss /= len(self.train_loader)
            epoch_train_errors.append(train_loss)
            val_loss = self.evaluate(model, criterion)
            epoch_val_errors.append(val_loss)
            # Guardar los pesos del mejor modelo basado en la pérdida de validación
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())

            early_stopping(val_loss)
                

            if self.log_fn is not None:
                if (epoch + 1) % log_every == 0:
                    self.log_fn(epoch, train_loss, val_loss)

            if do_early_stopping and early_stopping.early_stop:
                print(
                    f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f}"
                )
                break

        # Cargar los mejores pesos al modelo al final del entrenamiento
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        return model, epoch_train_errors, epoch_val_errors
        
    def inference(self):
        print("Inferencing the model")
        
    def __str__(self):
        return "BinPackingProblemSolver"