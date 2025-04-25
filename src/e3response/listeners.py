import copy
import logging
import sys
from typing import Union

import jax
import reax
from reax.listeners import utils
import tqdm
from typing_extensions import override

_LOGGER = logging.getLogger(__name__)


class E3mdPrinter(reax.TrainerListener):
    """
    See https://docs.python.org/3/library/stdtypes.html#old-string-formatting for formatting style
    """

    _progress_bar = None
    COLUMN_WIDTH = 10
    NUMBER_DECIMALS = 5

    def __init__(
        self,
        metrics: list[str],
        log_level=logging.INFO,
        log_every: int = 10,
    ):
        # Params
        self._log_level = log_level
        self._log_every = log_every

        header = ["epoch", "updates"]
        line = ["%(epoch)5i", "%(update)7i"]
        names = ["epoch", "update"]
        for entry in metrics:
            for stage in ("train", "val"):
                name = f"{stage}.{entry}"
                names.append(name)
                header.append(name[: self.COLUMN_WIDTH].center(self.COLUMN_WIDTH))
                line.append(f"%({name}){self.COLUMN_WIDTH}.{self.NUMBER_DECIMALS}f")

        self._metrics = names
        self._header = " ".join(header)
        self._line = line

    def init_train_tqdm(self, stage: "reax.stages.EpochStage") -> tqdm.tqdm:
        """Override this to customize the tqdm bar for training."""
        return tqdm.tqdm(
            desc=stage.name,
            # position=(2 * self.process_position),
            # disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            # bar_format=self.BAR_FORMAT,
        )

    @override
    def on_stage_starting(self, trainer: "reax.Trainer", stage: "reax.stages.Stage") -> None:
        if isinstance(stage, reax.stages.Fit):
            _LOGGER.log(self._log_level, self._header)
        if isinstance(stage, reax.stages.EpochStage):
            if self._progress_bar is not None:
                self._progress_bar.close()
            self._progress_bar = self.init_train_tqdm(stage)
            self._progress_bar.reset(total=utils.convert_inf(stage.max_iters))
            self._progress_bar.initial = 0

    @override
    def on_stage_ending(self, trainer: "reax.Trainer", stage: "reax.Stage") -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()

        if isinstance(stage, reax.stages.FitEpoch):
            if trainer.current_epoch % self._log_every == 0:
                msg, values = self._get_metrics(trainer)
                _LOGGER.log(self._log_level, msg, values)

    @override
    def on_stage_iter_starting(
        self, trainer: "reax.Trainer", stage: "reax.Stage", step: int
    ) -> None:
        if isinstance(stage, reax.stages.EpochStage) and self._progress_bar is not None:
            bar = self._progress_bar
            n = stage.iteration + 1
            bar.n = n
            bar.refresh()

    def _get_metrics(
        self, trainer: "reax.Trainer"
    ) -> tuple[str, dict[str, Union[jax.Array, "str"]]]:
        metrics = copy.deepcopy(trainer.progress_bar_metrics)
        metrics.setdefault("epoch", trainer.current_epoch)
        metrics.setdefault("update", trainer.global_updates)

        line = []
        for entry, fmt in zip(self._metrics, self._line):
            if entry in metrics:
                line.append(fmt)
            else:
                # Missing
                line.append("[n/a]".center(self.COLUMN_WIDTH))

        return " ".join(line), metrics
