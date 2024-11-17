import yaml


# Load yaml file function
def load_yaml(file_path):
    with open(file_path, "r") as file:
        value = yaml.safe_load(file)
    return value
