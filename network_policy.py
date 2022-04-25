import torch 
from torch import nn


# Implement CNN for policy parametrization


class PolicyNet(nn.Module):
    def __init__(self, dimension, action_dim):
        super().__init__()
        # Create the required layers/function and store them as instance variables
        self.dimension = dimension
        self.action_dim = action_dim
        self.layer1 = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(1, 4, kernel_size = 3,stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.fc1 = torch.nn.Linear(self.dimension*self.dimension*4, out_features=self.action_dim, bias=True)
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        """
        Perform the forward pass.

        Parameters
        ----------
        x: tensor of shape (batch_size, 1, state_dim, state_dim)

        Returns
        -------
        model output as a tensor of shape (batch_size, action_dim)
        """
        out = None

        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.log_softmax(out)


        return out