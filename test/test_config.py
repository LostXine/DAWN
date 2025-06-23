import sys
import hydra
from omegaconf import OmegaConf

if __name__ == "__main__":
    with hydra.initialize(config_path="configs"):
        argv = sys.argv[1:]
        print(argv)
        if len(argv) > 0 and argv[0].startswith("config="):
            config_name = argv[0].split("=")[1]
            argv = argv[1:]
        else:
            config_name = "default"
        cfg = hydra.compose(config_name=config_name, overrides=argv)
        OmegaConf.resolve(cfg)
        print("Configuration:\n" + OmegaConf.to_yaml(cfg))

