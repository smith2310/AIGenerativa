import torch
import torch.nn as nn
import torch.nn.functional as F


class BinPackingLSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, max_len=100):
        super(BinPackingLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.max_len = max_len
    
    def forward(self, x):
        # x: (batch_size, input_size)
        x = x.unsqueeze(1)
        # x: (batch_size, 1, input_size) to match the input shape of LSTM
        print(f'{x.shape=}')
        _, (hidden, cell) = self.encoder(x)

        print(f'{hidden.shape=}')
        print(f'{cell.shape=}')
        
        decoder_input = torch.zeros(x.size(0), 1, self.hidden_size).to(x.device)
        output = []

        for i in range(self.max_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_input = decoder_output
            output.append(decoder_output)
        
        output = torch.cat(output, dim=1)
        output = self.fc(output)
        return output