import yaml
import sys

with open("config.yaml", "r") as f:
    try:
        settings = yaml.safe_load(f)
        helper = settings['helper_functions']

    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)

