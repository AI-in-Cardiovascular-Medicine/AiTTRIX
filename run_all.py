import os

for config_name in ["rsf", "nac"]:
    print("\nRunning config:", config_name)
    cmd = f'python train_config.py --config-name {config_name}'
    os.system(cmd)
