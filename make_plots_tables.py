from hydra import initialize, compose
from report.make_tables import make_tables
from report.make_figures import MakeFigures


if __name__ == "__main__":
    # Combine ML with scores results, and make plots
    with initialize(version_base=None, config_path="config_files"):
        cfg = compose(config_name="rsf")
        MakeFigures(cfg)()
    # Make formatted tables
    make_tables()
