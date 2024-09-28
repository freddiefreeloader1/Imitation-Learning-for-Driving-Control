import torch.nn as nn
import torch

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_weights=[0.5, 1.5, 1.0, 2.0, 1.5, 0.3]):
        super(SimpleNet, self).__init__()
        self.input_weights = nn.Parameter(torch.tensor(input_weights), requires_grad=False)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.input_weights * x

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)     
        return x