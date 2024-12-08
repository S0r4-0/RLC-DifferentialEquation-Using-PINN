import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class FCN(nn.Module):
    "Defines a fully connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def rlc_circuit(R, L, C, t):
    """Defines the analytical solution to the underdamped RLC circuit problem.
    The current across the capacitor I(t) = I0 * exp(-at) * sin(ω_d * t + φ).
    Here, a is the damping ratio and ω_d is the damped natural frequency."""
    a = R / (2 * L)
    ω_0 = 1 / np.sqrt(L * C)
    ω_d = np.sqrt(ω_0**2 - a**2)
    I0 = 1
    phi = np.pi / 2
    
    return I0 * torch.exp(-a * t) * torch.sin(ω_d * t + phi)


def plot_result(t, I, t_data, I_data, Ih, step, tp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
    plt.plot(t,I, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(t,Ih, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(t_data, I_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if tp is not None:
        plt.scatter(tp, -0*torch.ones_like(tp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    plt.legend()
    plt.text(1.065,0.7,"Training step: %i"% step,fontsize="xx-large",color="k")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (I)")
    plt.title("RLC Circuit Current Response")
    plt.axis("off")

# RLC circuit parameters
R, L, C = 2.0, 1.0, 0.5

# Generate dataset
t = torch.linspace(0, 10, 500).view(-1, 1)
I = rlc_circuit(R, L, C, t).view(-1, 1)

# Training data
t_data = t[0:100:20]
I_data = I[0:100:20]