'''Dropout Module
-Author : Huicheol Moon
'''

import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract


class Dropout(nn.Module):
    """Dropout module."""

    def __init__(self, prob: int =0.5):
        """
        Args:
            prob: dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.dropout(x)


class DropoutGenerator(GeneratorAbstract):
    """Dropout module generator for parsing."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        p = self.args[0]
        if repeat > 1:
            module = []
            for i in range(repeat):
                module.append(Dropout(prob=p))
        else:
            module = Dropout(prob=p)
        return self._get_module(module)
