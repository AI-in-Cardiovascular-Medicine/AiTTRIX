import subprocess
import sys

for config_name in ["rsf", "nac"]:
    print("\nRunning config:", config_name)
    subprocess.run([sys.executable, "train_config.py", f"--config-name={config_name}"], check=True)

print("\nMake plots and tables from results")
subprocess.run([sys.executable, "make_plots_tables.py"], check=True)