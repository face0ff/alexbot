import torch
import torch.nn as nn
import numpy as np

class SMCTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(SMCTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Берем среднее по всей последовательности (Global Average Pooling) 
        # или только последний токен
        x = torch.mean(x, dim=1) 
        
        return self.classifier(x)
