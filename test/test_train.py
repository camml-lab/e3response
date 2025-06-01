import pathlib

import hydra
import pytest
import reax
import tensorial

import e3response
import e3response.data

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs"
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("model", ["nequip", "mace"])
def test_load_train(model):
    trainer = reax.Trainer()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        cfg = hydra.compose(config_name="train", overrides=[f"model={model}", "data=bto"])
        datamodule = e3response.data.BtoDataModule(r_max=5.0, data_dir=DATA_PATH / "bto")
        from_data = tensorial.config.FromData(
            cfg["from_data"],
            trainer.strategy,
            trainer.rng,
            datamodule=datamodule,
        )
        trainer._run_stage(from_data)

        module = tensorial.config.instantiate(cfg["model"])
        assert isinstance(module, tensorial.ReaxModule)
