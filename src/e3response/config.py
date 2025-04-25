import dataclasses
from typing import Any, Optional, Union

from flax.training import orbax_utils
import flax.training.train_state
import omegaconf
import orbax.checkpoint
import tensorial

__all__ = "load_module_state", "Initialisable"

Config = Union[omegaconf.DictConfig, omegaconf.ListConfig]

MODULE_STATE = "state"
MODULE_CONFIG = "config"
TRAIN_STATE = "train_state"

DEFAULT_CONFIG_FILE = "config.yaml"
MODULE_PARAMS_FILENAME = "params.ckpt"


@dataclasses.dataclass
class FromData:
    n_elements: Optional[int]
    atomic_numbers: Optional[list[int]]
    avg_num_neighbours: Optional[float]


@dataclasses.dataclass
class Globals:
    r_max: float
    rng_seed: int = 0
    accelerator: Optional[str] = None


Initialisable = dict


@dataclasses.dataclass
class Training:
    min_epochs: Optional[int]
    loss_fn: Initialisable
    optimiser: Initialisable
    metrics_registry: Optional[dict[str, Initialisable]]
    metrics: dict[str, str]
    datasets: Initialisable

    max_epochs: Optional[int] = tensorial.training.DEFAULT_MAX_EPOCHS
    batch_size: Optional[int] = 16
    shuffle: Optional[bool] = True
    shuffle_every: Optional[int] = 1


def create_train_checkpoint(train_state: flax.training.train_state.TrainState):
    return {
        TRAIN_STATE: train_state,
    }


def create_module_checkpoint(module_config: Config, module_state) -> dict:
    return {
        MODULE_CONFIG: omegaconf.OmegaConf.to_container(module_config, resolve=True),
        MODULE_STATE: module_state,
    }


def save_module(path, module_config: Config, module_state):
    save_state = create_module_checkpoint(module_config, module_state)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(
        path,
        save_state,
        save_args=orbax_utils.save_args_from_target(save_state),
    )


def load_module_state(path) -> tuple[Config, Any]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore(path)
    return omegaconf.OmegaConf.create(state[MODULE_CONFIG]), state[MODULE_STATE]
