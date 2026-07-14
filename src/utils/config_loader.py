import yaml


def load_config(path):

    with open(path) as file:
        return yaml.safe_load(file)