import pickle  # nosec
from typing import Any

import hydra
from omegaconf import DictConfig
import reax

from . import utils

log = utils.RankedLogger(__name__, rank_zero_only=True)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # pylint: disable=protected-access
    datamodule: reax.DataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # pylint: disable=protected-access
    trainer: reax.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    with open(cfg.ckpt_path, "rb") as file:
        model: reax.Module = pickle.load(file)  # nosec

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, **cfg.get("train", {}))

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.listener_metrics

    return metric_dict, object_dict


def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    runner = hydra.main(
        version_base="1.3",
        config_path="../../configs",
        config_name="eval.yaml",
    )(main)

    runner()
