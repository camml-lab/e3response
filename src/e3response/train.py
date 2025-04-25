import logging
import pathlib
from typing import Optional

import hydra
from hydra.core import hydra_config
import jraph
import omegaconf
import reax.utils

from . import config, utils

_LOGGER = logging.getLogger(__name__)

GraphsData = tuple[jraph.GraphsTuple]


def train(cfg: omegaconf.DictConfig):
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)

    # set seed for random number generators in JAX, numpy and python.random
    if cfg.get("seed"):
        reax.seed_everything(cfg.seed, workers=True)

    _LOGGER.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: reax.DataModule = hydra.utils.instantiate(cfg.data, _convert_="object")

    _LOGGER.info("Instantiating listeners...")
    listeners: list[reax.TrainerListener] = utils.instantiate_listeners(cfg.get("listeners"))

    _LOGGER.info("Instantiating loggers...")
    logger: list[reax.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    _LOGGER.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: reax.Trainer = hydra.utils.instantiate(cfg.trainer, listeners=listeners, logger=logger)

    if cfg.get("from_data"):
        trainer.run(
            utils.from_data.FromData(
                cfg["from_data"], trainer.strategy, trainer.rng, datamodule=datamodule
            )
        )

    _LOGGER.info(f"Instantiating model <{cfg.model._target_}>")
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

    # Save the configuration file here, this way things like inputs used to setup the model
    # will be baked into the input
    with open(output_dir / config.DEFAULT_CONFIG_FILE, "w") as file:
        file.write(omegaconf.OmegaConf.to_yaml(cfg))

    # bounds = (float("inf"), -float("inf"))
    # for name, loader in [
    #     ("train", data_module.train_dataloader()),
    #     ("validate", data_module.val_dataloader()),
    # ]:
    #     labels = []
    #     predictions = []
    #     for entry in trainer.predict(mod, dataloaders=loader):
    #         batch = jraph.unpad_with_graphs(entry)
    #         energies = batch.globals["energy"][:, 0] / batch.n_node
    #         labels.extend(energies.tolist())
    #
    #         predicted_energies = batch.globals["predicted_energy"].array[:, 0] / batch.n_node
    #         predictions.extend(predicted_energies.tolist())
    #
    #     x = np.array(labels)
    #     y = np.array(predictions)
    #
    #     bounds = (
    #         min(x.min(), y.min(), bounds[0]) - int(0.1 * y.min()),
    #         max(x.max(), y.max(), bounds[1]) + int(0.1 * y.max()),
    #     )
    #     ax = plt.gca()
    #
    #     # Ensure the aspect ratio is square
    #     ax.set_aspect("equal", adjustable="box")
    #     plt.plot(x, y, "o", alpha=0.5, ms=10, markeredgewidth=0.0, label=name)
    #
    # ax.set_xlim(bounds)
    # ax.set_ylim(bounds)
    # ax.legend()
    # plt.savefig(os.path.join(output_dir, "parity.pdf"), bbox_inches="tight")

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
        _LOGGER.info(f"Best ckpt path: {ckpt_path}")

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
