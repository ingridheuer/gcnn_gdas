import pathlib
import tomli

path = pathlib.Path(__file__).parent / "nlp_args.toml"
with path.open(mode="rb") as fp:
    nlp_args = tomli.load(fp)