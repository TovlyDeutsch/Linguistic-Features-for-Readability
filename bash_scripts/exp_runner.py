import os
import yaml
import sys


def run_exp(dir: str):
  for dirpath, dirnames, filenames in os.walk(dir):
    for filename in filenames:
      if filename.endswith('.yaml'):
        filepath = os.path.join(dirpath, filename)
        config = yaml.load(open(filepath).read())
        if 'k_fold' in config and 'subsample_splits' in config and config[
                'k_fold'] > 0 and config['subsample_splits'] >= 1 and config.get('run', True):
          start_array_num = 0
          end_array_num = config['k_fold'] * config['subsample_splits'] - 1
          print(
              f"Running {filepath} from {start_array_num} to {end_array_num} with {config['runner_type']}")
          if config['runner_type'] == 'cpu_single':
            os.system(
                f'bash bash_scripts/run_config_single_cpu.sh {filepath} {start_array_num} {end_array_num}')
          elif config['runner_type'] == 'cpu_multi':
            os.system(
                f'bash bash_scripts/run_config_multi_cpu.sh {filepath} {start_array_num} {end_array_num}')
          else:
            os.system(
                f'bash bash_scripts/run_config_gpu.sh {filepath} {start_array_num} {end_array_num}')


if __name__ == "__main__":
  run_exp('configs/WeeBit')
