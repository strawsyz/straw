class ConfigChecker:
    def __init__(self):
        super(ConfigChecker, self).__init__()

    # def show_config(self):
    #     from prettytable import PrettyTable
    #     config_view = PrettyTable()
    #     config_view.field_names = ["name", "value"]
    #     for attr in self.__dict__:
    #         config_view.add_row([attr, getattr(self, attr)])
    #     return config_view
    #     return "\n{}".format(config_view)

    def check_config(self):
        if getattr(self, "needed_config", None) is not None:
            for needed_attr in self.needed_config.split(" "):
                if getattr(self, needed_attr, None) is None:
                    self.logger.error("not set {} in the config file for {}".format(needed_attr, __file__))
