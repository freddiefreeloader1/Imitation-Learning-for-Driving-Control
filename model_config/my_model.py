import torch.nn as nn
import torch
import torch.optim as optim
import math



class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
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

class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.model = model
        self.betas = self._linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        return alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise

    def loss(self, x_start):
        noise = torch.randn_like(x_start)
        t = torch.randint(0, self.timesteps, (x_start.shape[0],))
        x_noisy = self.forward(x_start, t)
        predicted_noise = self.model(x_noisy)
        return F.mse_loss(predicted_noise, noise)

    def sample(self, shape):
        x = torch.randn(shape)
        for t in reversed(range(self.timesteps)):
            alpha_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            z = torch.randn_like(x) if t > 0 else 0
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t).sqrt() * self.model(x)) + beta_t.sqrt() * z
        return x


