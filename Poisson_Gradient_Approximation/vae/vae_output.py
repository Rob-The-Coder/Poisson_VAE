import torch

from dataclasses import dataclass
from typing import Optional

@dataclass
class VAEOutput:
  reconstruction: torch.Tensor
  p1: torch.Tensor
  p2: Optional[torch.Tensor] = None