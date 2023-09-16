import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import utils

import os

import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info(utils.get_original_cwd())
    log.info(os.getcwd())
    
if __name__ == "__main__":
    main()
