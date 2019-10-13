# config_loader.py
import json
import os


class Config:
    data = {}
    configFilePath = (os.path.join(os.getcwd(), 'experiment_configurations/experiment.json'))
    with open(configFilePath) as config_file:
        data = json.load(config_file)

    @classmethod
    def get(cls, key):
        return cls.data[key]

