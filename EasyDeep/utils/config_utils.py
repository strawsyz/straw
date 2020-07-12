class ConfigChecker:
    def __init__(self):
        super(ConfigChecker, self).__init__()

    def list_config(self):
        configs = []
        for attr in self.__dict__:
            configs.append("{} \t {}".format(attr, getattr(self, attr)))
        return configs

    def check_config(self):
        if getattr(self, "needed_config", None) is not None:
            for needed_attr in self.needed_config.split(" "):
                if getattr(self, needed_attr, None) is None:
                    self.logger.error("not set {} in the config file for {}".format(needed_attr, __file__))
