import yaml


class Config(object):
    def __init__(self):
        with open("../config/config.yaml", "r", encoding="utf-8") as cf:
            self.config = yaml.safe_load(cf)

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance