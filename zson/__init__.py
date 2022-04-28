from typing import Dict
from src.policy import (
    Net, Policy,
)
from .zson_policy import (
    ZSONPolicy
)
from .sensors import ObjectGoalPromptSensor
POLICY_CLASSES = {
    'ZSONPolicy': ZSONPolicy,
}

__all__ = [
    "Policy", "Net",
]