from typing import Any, Optional

import clip
import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes


@registry.register_sensor
class ObjectGoalPromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoalprompt"

    def __init__(
        self,
        *args: Any,
        config: Config,
        **kwargs: Any,
    ):
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=np.inf, shape=(77,), dtype=np.int64)

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        # the attribute check below makes the sensor hackable
        cat = episode.object_category if hasattr(episode, "object_category") else ""
        prompt = self.config.PROMPT.format(cat)
        return clip.tokenize(prompt, context_length=77).numpy()
