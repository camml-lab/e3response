import collections
import logging
import pathlib
import pickle

import hydra
from hydra.core import hydra_config
import jax
import jax.random
import jraph
import omegaconf
import reax.utils
import tensorial
import yaml

from . import config, data, train, utils

_LOGGER = logging.getLogger(__name__)

MODEL_PARAMS = "model_params"


def predict(cfg: omegaconf.DictConfig):
    if MODEL_PARAMS not in cfg:
        # Taken from https://github.com/facebookresearch/hydra/discussions/2750
        hcfg = hydra_config.HydraConfig.get()
        config_path = [
            path["path"] for path in hcfg.runtime.config_sources if path["schema"] == "file"
        ][0]
        config_path = pathlib.Path(config_path)
        files = config_path.glob("*.ckpt")
        try:
            params_path = max(files, key=lambda path: path.stat().st_mtime)
        except ValueError:
            raise ValueError(f"Please provide module params using the '{MODEL_PARAMS}' key")
    else:
        params_path = pathlib.Path(cfg[MODEL_PARAMS])

    # Load the module parameters as our checkpoint
    if params_path.is_dir():
        params_path = params_path / config.MODULE_PARAMS_FILENAME
    if not params_path.exists():
        raise ValueError(f"'{params_path}' does not exist")
    with open(params_path, "rb") as file:
        params = pickle.load(file)

    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    # with open(output_dir / "config.yaml", "w") as file:
    #     file.write(omegaconf.OmegaConf.to_yaml(cfg))

    # Inform the user if an accelerator is being used
    _LOGGER.info("Using JAX backend: %s", cfg.globals.accelerator)

    # Load data
    datasets = hydra.utils.instantiate(cfg.predict.datasets)
    data_module = data.create_data_module(cfg.predict, datasets)

    # Create the module and load the parameters
    mod = tensorial.TrainingModule(cfg)
    mod.set_parameters(params)

    # Create trainer
    rng_key = jax.random.key(cfg.globals.rng_seed)
    reax.seed_everything(cfg.globals.rng_seed)
    trainer = reax.Trainer(
        mod,
        accelerator=cfg.globals.accelerator,
        rng_key=rng_key,
    )

    # Make the predictions
    res = trainer.predict(mod, datamodule=data_module, return_predictions=True)

    # Extract what we want from the outputs
    to_dump = list()
    for entry in res:
        batch = jraph.unpad_with_graphs(entry)
        for energy in batch.globals["predicted_energy"].array[:, 0].tolist():
            outputs = dict(predicted_energy=energy)
            to_dump.append(outputs)

    with open("predictions.yaml", "w", encoding="utf-8") as file:
        yaml.dump(to_dump, file, default_flow_style=False)
