#!python

import os
import yaml
import sys

# config = None
env = os.getenv('ENV', 'prod')
config_file_name = os.getenv('CONFIG', 'config.yml')
config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file_name)
with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)[env]

def custom_config(config_file_path, env):
    with open(config_file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)[env]


def get(key, default=None):
    if key in config:
        if config[key] is None:
            return default
        else:
            return config[key]
    else:
        return default


if __name__ == '__main__':
    print(get(sys.argv[1], ''))
