import torch

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VAEOutput:
  reconstruction: torch.Tensor | List[torch.Tensor]
  p1: torch.Tensor
  p2: Optional[torch.Tensor | List[torch.Tensor]] = None