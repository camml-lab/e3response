import logging
import pathlib
from typing import Optional

import hydra
from hydra.core import hydra_config
import jraph
import omegaconf
import reax.utils
import tensorial

from . import config, utils

_LOGGER = logging.getLogger(__name__)

GraphsData = tuple[jraph.GraphsTuple]


def train(cfg: omegaconf.DictConfig):
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)

    # set seed for random number generators in JAX, numpy and python.random
    if cfg.get("seed"):
        reax.seed_everything(cfg.seed, workers=True)

    _LOGGER.info(
        "Instantiating datamodule <%s>", cfg.data._target_  # pylint: disable=protected-access
    )
    datamodule: reax.DataModule = hydra.utils.instantiate(cfg.data, _convert_="object")

    _LOGGER.info("Instantiating listeners...")
    listeners: list[reax.TrainerListener] = utils.instantiate_listeners(cfg.get("listeners"))

    _LOGGER.info("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    _LOGGER.info(
        "Instantiating trainer <%s>", cfg.trainer._target_  # pylint: disable=protected-access
    )
    trainer: reax.Trainer = hydra.utils.instantiate(cfg.trainer, listeners=listeners, logger=logger)

    if cfg.get("from_data"):
        stage = trainer.run(
            tensorial.config.FromData(  # pylint: disable=no-member
                cfg["from_data"], trainer.strategy, trainer.rng, datamodule=datamodule
            )
        )
        _LOGGER.info(
            "Calculated from data (these can be used in your config files using "
            "${from_data.<name>}:\n%s",
            stage.calculated,
        )

    # Save the configuration file here, this way things like inputs used to setup the model
    # will be baked into the input
    with open(output_dir / config.DEFAULT_CONFIG_FILE, "w", encoding="utf-8") as file:
        file.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))

    _LOGGER.info("Instantiating model <%s>", cfg.model._target_)  # pylint: disable=protected-access
    model: reax.Module = hydra.utils.instantiate(cfg.model, _convert_="object")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "listeners": listeners,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _LOGGER.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Fit the potential
    if cfg.get("train"):
        _LOGGER.info("Starting training!")
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), **cfg.get("train")
        )

    train_metrics = trainer.listener_metrics

    if cfg.get("test"):
        _LOGGER.info("Starting testing!")
        ckpt_path = trainer.checkpoint_listener.best_model_path
        if ckpt_path == "":
            _LOGGER.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        _LOGGER.info("Best ckpt path: %s", ckpt_path)

    test_metrics = trainer.listener_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main(cfg: omegaconf.DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    runner = hydra.main(
        version_base="1.3",
        config_path="../../configs",
        config_name="train.yaml",
    )(main)
    runner()
