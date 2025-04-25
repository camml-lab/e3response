import collections
import os
from typing import Any, cast

import jraph
import matplotlib.pyplot as plt
import numpy as np
import reax
from typing_extensions import override

__all__ = ("Plotter",)


class StageListener(reax.stages.StageListener):
    def __init__(self, stage: reax.stages.EpochStage, output_path: str):
        # Params
        self._output_path = output_path
        # State
        self._stage: reax.stages.EpochStage = stage
        self._stage.events.add_listener(self)
        self._stats = collections.defaultdict(list)

    @override
    def on_stage_starting(self, stage: "reax.Stage", /):
        assert stage is self._stage
        self._stats = collections.defaultdict(list)

    @override
    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        assert stage is self._stage
        outputs = cast(jraph.GraphsTuple, outputs)
        jraph.unpad_with_graphs(outputs)
        batch_globals = outputs.globals
        if "energy" in batch_globals and "predicted_energy" in batch_globals:
            for label in ("energy", "predicted_energy"):
                self._stats[label].extend(batch_globals[label].array[:, 0].tolist())

    @override
    def on_stage_ending(self, stage: "reax.Stage", /):
        assert stage is self._stage
        x = np.array(self._stats["energy"])
        y = np.array(self._stats["predicted_energy"])
        if x and y:
            bounds = (
                min(x.min(), y.min()) - int(0.1 * y.min()),
                max(x.max(), y.max()) + int(0.1 * y.max()),
            )
            ax = plt.gca()
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            # Ensure the aspect ratio is square
            ax.set_aspect("equal", adjustable="box")
            plt.plot(x, y, "o", alpha=0.5, ms=10, markeredgewidth=0.0)
            plt.savefig(os.path.join(self._output_path, "parity.pdf"), bbox_inches="tight")


class Plotter(reax.TrainerListener):
    def __init__(self, output_path: str):
        self._training = None
        self._validation = None
        self._output_path = output_path

    @override
    def on_train_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train") -> None:
        if self._training is None:
            self._training = StageListener(stage, self._output_path)

    @override
    def on_validation_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        if self._validation is None:
            self._validation = StageListener(stage, self._output_path)
