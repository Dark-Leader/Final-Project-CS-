import yaml
import sys

with open("config.yaml", "r") as f:
    try:
        settings = yaml.safe_load(f)
        helper = settings['helper_functions']
        pre = settings['preprocessing']
        st = settings['staff']
        coor = settings['coordinator']

    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)

