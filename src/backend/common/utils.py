import yaml
import json


# Load yaml file function
def load_yaml(file_path):
    with open(file_path, "r") as file:
        value = yaml.safe_load(file)
    return value


# Load JSON file
def load_json(file_path):
    with open(file_path, "r") as file:
        value = json.load(file)
    return value
