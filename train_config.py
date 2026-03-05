import os
import sys

import hydra
from omegaconf import OmegaConf
import enlighten
import numpy as np
from loguru import logger

from preprocessing.preprocessing import Preprocessing
from survival.survival_experiment import SurvivalExperiment
from explainability.shap_runner import RunExplainability


@hydra.main(version_base=None, config_path="config_files", config_name="rsf")
def main(config):
    if config.meta.out_dir is None:
        config.meta.out_dir = os.path.splitext(config.meta.train_file)[0]
    os.makedirs(config.meta.out_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.meta.out_dir, "config.yaml"))
    logfile_path = os.path.join(config.meta.out_dir, "logs.log")
    file = open(logfile_path, 'a')
    sys.stderr = file
    sys.stdout = file
    logger.add(logfile_path, mode='a')
    progress_manager = enlighten.get_manager()
    pbar = progress_manager.counter(total=config.meta.n_seeds, desc='Seeds', unit='seeds')
    seed = config.meta.seed
    np.random.seed(config.meta.seed)
    preprocessing = Preprocessing(config)
    pipeline = SurvivalExperiment(config, progress_manager)

    # Train model
    np.random.seed(seed)
    data_x_train, data_x_test, data_y_train, data_y_test = preprocessing(seed)
    logger.info(f"Preprocessing for seed {seed} is finished")
    _ = pipeline(
        seed,
        data_x_train,
        data_y_train,
        data_x_test,
        data_y_test
    )
    pbar.update()
    pbar.close()

    if config.explainability.compute_shap:
        logger.info(f"Computing SHAP values...")
        RunExplainability(config=config).run()


if __name__ == "__main__":
    main()
