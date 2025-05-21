from typing import Union

import omegaconf

Config = Union[omegaconf.DictConfig, omegaconf.ListConfig]

MODULE_STATE = "state"
MODULE_CONFIG = "config"
TRAIN_STATE = "train_state"

DEFAULT_CONFIG_FILE = "config.yaml"
MODULE_PARAMS_FILENAME = "params.ckpt"
