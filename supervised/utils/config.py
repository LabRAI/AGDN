from argparse import Namespace

import yaml


def load_config(mode='train'):
    config_path = './configs/'
    config_name = f'{mode}.yml'
    config_default_name = 'default.yml'

    # Load Config
    with open(config_path + config_name, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Namespace(**config_dict)

    # Load Default Config
    with open(config_path + config_default_name, 'r') as f:
        default_dict = yaml.safe_load(f)

    for k, v in default_dict.items():
        if not hasattr(config, k):
            setattr(config, k, v)

    # print
    print(config)

    return config


if __name__ == '__main__':
    config = load_config()
    print(config)
