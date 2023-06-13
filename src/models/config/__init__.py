import pathlib
import tomli

path = pathlib.Path(__file__).parent / "train_config.toml"
with path.open(mode="rb") as fp:
    train_config = tomli.load(fp)