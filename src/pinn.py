import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NavierStokesNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden_layer1 = nn.Linear(3,20) # 3 input features: x, y, t
        self.hidden_layer2 = nn.Linear(20,20)
        self.hidden_layer3 = nn.Linear(20,20)
        self.hidden_layer4 = nn.Linear(20,20)
        self.hidden_layer5 = nn.Linear(20,20)
        self.hidden_layer6 = nn.Linear(20,20)
        self.hidden_layer7 = nn.Linear(20,20)
        self.hidden_layer8 = nn.Linear(20,20)
        self.hidden_layer9 = nn.Linear(20,20)
        self.output_layer = nn.Linear(20,2) # 2 output features: u, v (velocity, pressure)
    
    def forward(self, x):
        pass

def physics_informed_loss(model, x, y ,t):
    pass