import pathlib
import tomli

path = pathlib.Path(__file__).parent / "viz_config.toml"
with path.open(mode="rb") as fp:
    viz_config = tomli.load(fp)