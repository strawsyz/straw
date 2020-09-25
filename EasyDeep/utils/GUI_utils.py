import os
from tkinter import Button, Entry, Frame, Label, StringVar, Tk, Toplevel
from tkinter import messagebox
from tkinter import ttk
from tkinter.ttk import Notebook

import torch

from configs import GUI_config


class Application_ui(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('ConfigGUI')
        self.master.geometry('1068x681+50+50')
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        self.main_strip = Notebook(self.top)
        self.main_strip.place(relx=0.062, rely=0.071, relwidth=0.887, relheight=0.876)

        self.experiment_tab = Frame(self.main_strip)
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

        self.hidden_attr_names = ["logger"]
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
        """get value of config file to display"""
        value = getattr(config, attr_name)
        if "record" in attr_name:
            value = value.get_class_name()
        return value

    def init_tab_window(self, tab_type):
        config_name = "{}_config".format(tab_type)
        config = getattr(self, config_name)
        tab_window_name = "{}_tab".format(tab_type)
        tab_window = getattr(self, tab_window_name)
        idx = 0
        for idx, (attr, value) in enumerate(config.get_attrs_4_gui()):
            if idx % 2 == 0:
                label_bg = "#48D1CC"
                input_bg = ""
            else:
                label_bg = "#d0af4c"
                input_bg = ""
            if attr in self.hidden_attr_names:
                continue
            # value = self.get_value_by_attr_name(config, attr)

            label_name = attr + "_label"
            label = Label(tab_window, text=attr, width=50, bg=label_bg)
            input_name = attr + "_text"
            setattr(self, label_name, label)
            label.grid(row=idx, column=0, sticky='e')
            label.description = "モデルを保存するフォルダのパス"
            label.bind("<Enter>", self.add_tip_window)
            label.bind("<Leave>", self.destroy_tip_window)
            label.bind("<ButtonPress>", self.destroy_tip_window)
            if attr in self.read_only_attr_names:
                input = Label(tab_window, text=value, bg=label_bg)
            else:
                if isinstance(value, bool):
                    if value is True:
                        combobox_idx = 0
                    else:
                        combobox_idx = 1
                    options_name = attr + "_option"
                    options = StringVar()
                    setattr(self, options_name, options)
                    input = ttk.Combobox(tab_window, width=17, textvariable=options)
                    input['values'] = ("はい", "いいえ")
                    input.current(combobox_idx)
                else:
                    var = StringVar()
                    text_name = attr + "_text"
                    setattr(self, text_name, var)
                    input = Entry(tab_window, textvariable=var, width=max(20, len(str(value))))
                    var.set(value)

            setattr(self, input_name, input)
            input.grid(row=idx, column=1, sticky='w')

        save_button_name = "{}_save_button".format(tab_type)
        save_button = Button(tab_window, text="保存", bg="lightblue", width=10,
                             command=lambda: self.save_config(tab_type))
        setattr(self, save_button_name, save_button)
        save_button.grid(row=idx + 1, column=1)

    def init_all_tab_windows(self):
        self.init_tab_window("experiment")
        self.init_tab_window("dataset")
        self.init_tab_window("net")

    def add_tip_window(self, event):
        """
        create a tip window
        """
        widget = event.widget
        params = {
            'text': widget.description,
            'justify': 'left',
            'background': "lightyellow",
            'relief': 'solid',
            'borderwidth': 1
        }
        self.tip_window = self.create_tip_window(widget, event)
        label = ttk.Label(self.tip_window, **params)
        label.grid(sticky='nsew')

    def create_tip_window(self, widget, event):
        window = Toplevel(widget)
        window.overrideredirect(True)
        window.attributes("-toolwindow", 1)
        window.attributes("-alpha", 0.9)
        x = widget.winfo_rootx() + event.x
        y = widget.winfo_rooty() + event.y
        window.wm_geometry("+%d+%d" % (x, y))
        return window

    def destroy_tip_window(self, event):
        """
        destroy tooltip window
        """
        if self.tip_window is not None:
            self.tip_window.destroy()

    def save_config(self, config_type):
        """save config file"""
        config_name = "{}_config".format(config_type)
        config = getattr(self, config_name)
        for attr in config.__dict__:
            input_name = attr + "_text"
            input = getattr(self, input_name)
            if isinstance(input, Entry):
                new_value = input.get().strip()
                if new_value == "はい":
                    new_value = True
                elif new_value == "いいえ":
                    new_value = False
                setattr(config, attr, new_value)
        torch.save(config, self.experiment_config_save_path)
        messagebox.showinfo("message", 'save successfully')

    def load_config(self, config_save_path: str):
        """load config from disk"""
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
    ZMJ_PORTAL.init_all_tab_windows()
    init_window.mainloop()


if __name__ == '__main__':
    main()
