from tkinter import ttk
import pickle
from configs import GUI_config
import os
# from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Notebook
from tkinter import Button, Entry, Frame, Label, StringVar, Tk


class Application_ui(Frame):
    # 这个类仅实现界面生成功能，具体事件处理代码在子类Application中。
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('ConfigGUI')
        self.master.geometry('1068x681+50+50')
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        # self.style = Style()

        self.main_strip = Notebook(self.top)
        self.main_strip.place(relx=0.062, rely=0.071, relwidth=0.887, relheight=0.876)

        self.experiment_tab = Frame(self.main_strip)
        # self.TabStrip1__Tab1Lbl = Label(self.TabStrip1__Tab1, text='Please add widgets in code.')
        # self.TabStrip1__Tab1Lbl.place(relx=0.1, rely=0.5)
        self.main_strip.add(self.experiment_tab, text='Experiment')

        self.net_tab = Frame(self.main_strip)
        self.main_strip.add(self.net_tab, text='Net')

        self.dataset_tab = Frame(self.main_strip)
        self.main_strip.add(self.dataset_tab, text='Dataset')


class ConfigGUI(Application_ui):
    def __init__(self, experiment_config_save_path, dataset_config_save_path=None,
                 net_config_save_path=None, experiment_config=None, net_config=None, data_config=None):
        super(ConfigGUI, self).__init__()
        self.init_window_name = self.experiment_tab
        self.experiment_config_save_path = experiment_config_save_path
        self.dataset_config_save_path = dataset_config_save_path
        self.net_config_save_path = net_config_save_path
        self.set_config("experiment", experiment_config_save_path, experiment_config)
        self.set_config("dataset", dataset_config_save_path, data_config)
        self.set_config("net", net_config_save_path, net_config)

        self.read_only_attr_names = ["experiment_record", "recorder", "_system"]

    def set_config(self, config_type, config_save_path: str, config_instance=None):
        save_path_name = "{}_config_save_path".format(config_type)
        config_name = "{}_config".format(config_type)
        setattr(self, save_path_name, config_save_path)
        if config_save_path is not None and os.path.exists(config_save_path) and os.path.isfile(config_save_path):
            setattr(self, config_name, self.load_config(config_save_path))
        else:
            if config_instance is not None:
                setattr(self, config_name, config_instance)
            else:
                print("can't find config file for {}".format(config_type))

    def get_value_by_attr_name(self, config, attr_name):
        value = getattr(config, attr_name)
        if "record" in attr_name:
            value = value.get_class_name()
        return value

    def init_window(self, tab_type):
        config_name = "{}_config".format(tab_type)
        config = getattr(self, config_name)
        tab_window_name = "{}_tab".format(tab_type)
        tab_window = getattr(self, tab_window_name)

        for idx, attr in enumerate(config.__dict__):
            value = self.get_value_by_attr_name(config, attr)

            label_name = attr + "_label"
            label = Label(tab_window, text=attr, width=50)
            input_name = attr + "_text"
            setattr(self, label_name, label)
            label.grid(row=idx, column=0)
            if attr in self.read_only_attr_names:
                input = Label(tab_window, text=value)
            else:
                if isinstance(value, bool):
                    if value is True:
                        combobox_idx = 0
                    else:
                        combobox_idx = 1
                    options_name = attr + "_option"
                    options = StringVar()
                    setattr(self, options_name, options)
                    input = ttk.Combobox(tab_window, width=15, textvariable=options)
                    input['values'] = ("はい", "いいえ")
                    input.current(combobox_idx)
                else:
                    var = StringVar()
                    text_name = attr + "_text"
                    setattr(self, text_name, var)
                    input = Entry(tab_window, textvariable=var, width=max(20, len(str(value))))
                    var.set(value)
                    print(value)

            setattr(self, input_name, input)
            input.grid(row=idx, column=1)

        self.save_button = Button(tab_window, text="保存", bg="lightblue", width=10,
                                  command=self.save_config)
        self.save_button.grid(row=idx + 1, column=1)

    def set_init_window(self):
        # self.init_window_name["bg"] = "gray"
        # self.init_window_name.attributes("-alpha",0.9)
        self.init_window("experiment")
        self.init_window("dataset")
        self.init_window("net")

    def save_config(self):
        for attr in self.experiment_config.__dict__:
            input_name = attr + "_text"
            input = getattr(self, input_name)
            if isinstance(input, Entry):
                new_value = input.get().strip()
                setattr(self.experiment_config, attr, new_value)
        import torch
        torch.save(self.experiment_config, self.experiment_config_save_path)
        messagebox.showinfo("message", 'save successfully')

    def load_config(self, config_save_path: str):
        import torch
        return torch.load(config_save_path)


def main():
    from configs.experiment_config import MnistConfig
    config = MnistConfig()
    net_config = config.net_config
    dataset_config = config.dataset_config
    init_window = Tk()
    ZMJ_PORTAL = ConfigGUI(GUI_config.experiment_config_path, GUI_config.dataset_config_path,
                           GUI_config.net_config_path, experiment_config=config, net_config=net_config,
                           data_config=dataset_config)
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()


if __name__ == '__main__':
    main()
