import hydra
import omegaconf
import reax

from . import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


__all__ = "instantiate_listeners", "instantiate_loggers"


def instantiate_listeners(
    listeners_cfg: omegaconf.DictConfig,
) -> list[reax.TrainerListener]:
    """Instantiates listeners from config.

    :param listeners_cfg: A DictConfig object containing listener configurations.
    :return: A list of instantiated listeners.
    """
    listeners: list[reax.TrainerListener] = []

    if not listeners_cfg:
        log.warning("No listener configs found! Skipping..")
        return listeners

    if not isinstance(listeners_cfg, omegaconf.DictConfig):
        raise TypeError("listeners config must be a DictConfig!")

    for _, cb_conf in listeners_cfg.items():
        if isinstance(cb_conf, omegaconf.DictConfig) and "_target_" in cb_conf:
            log.info(
                f"Instantiating listener <{cb_conf._target_}>"  # pylint: disable=protected-access
            )
            listeners.append(hydra.utils.instantiate(cb_conf))

    return listeners


def instantiate_loggers(logger_cfg: omegaconf.DictConfig) -> list[reax.Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: list[reax.Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, omegaconf.DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, omegaconf.DictConfig) and "_target_" in lg_conf:
            log.info(
                f"Instantiating logger <{lg_conf._target_}>"  # pylint: disable=protected-access
            )
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
