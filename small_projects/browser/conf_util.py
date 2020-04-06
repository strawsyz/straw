import configparser


class ConfigureUtil():
    '''配置文件工具
    有读取、修改、增加、删除、保存功能
    增删改需要保存才会修改文件'''
    def __init__(self, path=None, encoding='utf-8'):
        self.config = configparser.ConfigParser()
        if not path:
            self.path = 'config.conf'            
            self.config.read(self.path, encoding=encoding)
        else:
            self.path = path
            self.config.read(path, encoding=encoding)

    def get(self, section, option, sort=None):
        res = self.config.get(section, option)
        if sort is None:
            return res
        if sort == 'int':
            return int(res)
        if sort == 'float':
            return float(res)
        return res

    def get_section(self, section):
        return self.config.items(section)

    def get_from_item(content, option):
        '''
        根据取出的item内容和option
        找到option对应的配置信息
        :param option:
        :return:
        '''
        for opt in content:
            if opt[0] == option:
                return opt[1]
        return option+'对应的配置信息'

    def add(self, section, key=None, value=None):
        '''
        key 和 value都不为None才会增加key-value对
        否则 只增加section
        :param section:
        :param key:
        :param value:
        :return:
        '''
        if key is None or value is None:
            return self.config.add_section(section)
        else:
            return self.config.set(section, key, value)

    def get_sections(self):
        # 获得配置文件中的对应的section
        return self.config.sections()

    def remove(self, section, option=None):
        if option is None:
            self.config.remove_section(section)
        else:
            self.config.remove_option(section, option)

    def save(self, mode='w'):
        '''
        保存对配置文件的增删改，默认重写模式
        :param mode:  a->追加模式 w->重新写入模式
        最好不用追加模式，会导致配置文件出现相同的section
        :return: 
        '''
        with open(self.path, mode) as file:
            self.config.write(file)



