import logging
from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from lumos.world_models.world_model import WorldModel

logger = logging.getLogger(__name__)


class CRSSMHighLevel(WorldModel):
    pass
