import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from abstract_bin_packing_solver import AbstractBinPackingSolver
from utils import EarlyStopping


class _BinPackingLanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout = 0, n_heads = 4):
        super(_BinPackingLanguageModel, self).__init__()
        # Embedding para las dimensiones de las cajas, (cada combinación posible de dimensiones, hidden_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        # Capa de salida para predecir las dimensiones de las cajas
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, container, target_seq = None, seq_len = 100):
        """
        Forward pass para entrenamiento o generación.
        
        Args:
            container: Tensor con las dimensiones del contenedor (batch_size, input_dim).
            target_seq: Tensor objetivo de las dimensiones de las cajas (batch_size, seq_len, output_dim).
            seq_len: Longitud máxima de la secuencia de salida.
        
        Returns:
            Salida generada o predicción durante el entrenamiento.
        """
        # container: (batch_size, input_dim)
        container_emb = self.embedding(container).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        if target_seq is not None:

            # Embedding del objetivo para entrenamiento
            target_emb = self.embedding(target_seq)  # (batch_size, seq_len, hidden_dim)
            tgt_len = target_emb.size(1)  # Longitud de la secuencia objetivo
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=container.device), diagonal=1).bool()
            output = self.transformer(
                src=container_emb,  # (batch_size, 1, hidden_dim)
                tgt=target_emb,     # (batch_size, seq_len, hidden_dim)
                tgt_mask=tgt_mask   # (seq_len, seq_len)
            )
            return self.fc_out(output)
        
        # Fase de generación
        generated_seq = container_emb  # Inicializamos la secuencia con el contenedor
        # generated_seq: (batch_size, 1, hidden_dim)
        for _ in range(seq_len):
            # Paso del Transformer para generar la siguiente caja
            transformer_out = self.transformer(generated_seq, generated_seq)
            
            # Tomamos la última salida para predecir la siguiente caja
            last_output = transformer_out[:, -1, :]  # (batch_size, hidden_dim)
            
            # Generamos la predicción de la caja
            predicted_box = self.fc_out(last_output)  # Predicción (width, height)

            #TODO: Cortar la secuencia si se predice un token con width o heigh igual a 0
            
            # Agregamos la caja generada a la secuencia para la próxima predicción
            predicted_box_emb = self.embedding(predicted_box)  # Embedding de la caja generada
            generated_seq = torch.cat([generated_seq, predicted_box_emb.unsqueeze(1)], dim=1)  # Agregar a la secuencia
        
        return generated_seq  # La secuencia generada de cajas


class BinPackingLanguageSolver(AbstractBinPackingSolver):
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
                output = model(x, y, seq_len)  # forward pass
                val_loss += criterion(output, y).item()  # acumulamos la perdida
        return val_loss / len(self.val_loader)  # retornamos la perdida promedio

    def train(self):
        # max_width = 50
        # max_height = 50
        model = _BinPackingLanguageModel(input_dim=2, hidden_dim=128, output_dim=2, n_layers=2, dropout=0.1).to(self.device)
        model.train()
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                y_pred = model(x, y, seq_len)
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