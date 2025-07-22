"""
Author: masato shibukawa
Date: 2025-07-22
Email: shibukawa.masato@ac.jaxa.jp

Physics-Informed Neural Networks (PINNs) implementation for neural particle methods.
Provides the base neural network architecture with automatic differentiation capabilities
for solving partial differential equations in fluid dynamics simulations.

Based on the PINNs framework by Maziar Raissi: https://github.com/maziarraissi/PINNs
"""

from torch import nn

class PhysicsInformedNNs(nn.Module):
    def __init__(self, layers, ub, lb, activation_func):
        super(PhysicsInformedNNs, self).__init__()
        self.linear_relu_stack = self.build_network(layers, activation_func)
        self.ub = ub
        self.lb = lb

    def build_network(self, layers, activation_func):
        layers_list = []
        for i in range(len(layers) - 1):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                layers_list.append(activation_func)
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        x = 2 * (x - self.lb) / (self.ub - self.lb) - 1
        return self.linear_relu_stack(x)