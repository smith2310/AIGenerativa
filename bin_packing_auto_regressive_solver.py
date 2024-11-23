import torch
from abstract_bin_packing_solver import AbstractBinPackingSolver
import torch.nn as nn


class BinPackingAutoRegressiveModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = torch.nn.Linear(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, container, target_seq = None, seq_len = 10):
        container_emb = self.embedding(container).unsqueeze(1)
        if target_seq is not None:
            # Entrenamiento: Concatenamos contenedor + secuencia objetivo
            target_emb = self.embedding(target_seq)
            lstm_input = torch.cat([container_emb, target_emb], dim=1)  # (batch_size, seq_len+1, hidden_dim)
            
            # Paso por el LSTM
            output, _ = self.lstm(lstm_input)
            return self.fc_out(output)  # (batch_size, seq_len+1, output_dim)

        # Generación autoregresiva
        generated_seq = container_emb
        hidden = None
        outputs = []
        
        for _ in range(seq_len):
            output, hidden = self.lstm(generated_seq, hidden)  # (batch_size, 1, hidden_dim)
            next_box = self.fc_out(output[:, -1, :])  # (batch_size, output_dim)
            outputs.append(next_box.unsqueeze(1))  # Guardamos la salida
            
            # Embedding del siguiente token generado
            next_box_emb = self.embedding(next_box)
            generated_seq = next_box_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_dim)
            
        

class BinPackingAutoRegressiveSolver(AbstractBinPackingSolver):
    def __init__(
            self,
            train_loader,
            val_loader,
            log_fn,
            device
        ):
        super().__init__(
            train_loader,
            val_loader,
            log_fn,
            device
        )
        
    def train(self):
        # Configuración básica
        input_dim = 2
        hidden_dim = 128
        output_dim = 2
        n_layers = 2
        dropout = 0.1

        # Inicializa el modelo
        model = BinPackingAutoRegressiveModel(input_dim, hidden_dim, output_dim, n_layers, dropout)
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 50

        # Entrenamiento
        for epoch in range(epochs):
            model.train()
            model.to(self.device)
            train_loss = 0
            
            for container, target_seq in self.train_loader:
                container = container.to(self.device)
                target_seq = target_seq.to(self.device)
                
                optimizer.zero_grad()
                output = model(container, target_seq)
                loss = criterion(output[:, 1:], target_seq)  # Excluimos la entrada inicial (contenedor)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(self.train_loader)}")
        return model


    def inference(self):
        pass