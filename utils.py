import os
import json


def read_json(file_path):
    if not os.path.isfile(file_path):
        raise ValueError("Config file is missing")
    if file_path.split(".")[-1].lower() != "json":
        raise ValueError("Invalid file format. Config should be a json")
    with open(file_path) as json_file:
        data = json.loads(json_file.read())
    return data
