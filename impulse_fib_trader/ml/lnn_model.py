import torch
import torch.nn as nn

class LiquidLayer(nn.Module):
    def __init__(self, in_features, hidden_features, dt=0.05):
        super(LiquidLayer, self).__init__()
        self.dt = dt 
        self.hidden_features = hidden_features

        self.W_in = nn.Linear(in_features, hidden_features)
        self.W_h = nn.Linear(hidden_features, hidden_features)
        self.tau = nn.Parameter(torch.rand(hidden_features) * 2.0 + 0.5)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_features, device=x.device)
        outputs = []

        for t in range(seq_len):
            forcing_term = torch.tanh(self.W_in(x[:, t, :]) + self.W_h(h))
            safe_tau = torch.clamp(self.tau, min=0.01) 
            dh = (-h / safe_tau) + forcing_term
            h = h + self.dt * dh
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)

class LiquidNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1):
        super(LiquidNet, self).__init__()
        self.liquid = LiquidLayer(in_features, hidden_features)
        self.readout = nn.Sequential(
            nn.Linear(hidden_features, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        liquid_states = self.liquid(x)
        # Возвращаем последнее состояние для классификации
        return self.readout(liquid_states[:, -1, :])
