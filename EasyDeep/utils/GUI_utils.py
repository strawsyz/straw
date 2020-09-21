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
    def __init__(self, init_window_name, config, config_save_path):
        super(ConfigGUI, self).__init__()
        self.init_window_name = self.experiment_tab
        self.config_save_path = config_save_path
        if os.path.exists(config_save_path) and os.path.isfile(config_save_path):
            self.load_config()
        else:
            self.config = config
        self.read_only_attr_names = ["experiment_record", "recorder", "_system"]

    def get_value_by_attr_name(self, attr_name):
        value = getattr(self.config, attr_name)
        if "record" in attr_name:
            value = value.get_class_name()
        return value

    def set_init_window(self):
        # self.init_window_name["bg"] = "gray"
        # self.init_window_name.attributes("-alpha",0.9)
        for idx, attr in enumerate(self.config.__dict__):
            value = self.get_value_by_attr_name(attr)

            label_name = attr + "_label"
            label = Label(self.init_window_name, text=attr)
            input_name = attr + "_text"
            setattr(self, label_name, label)
            label.grid(row=idx, column=0)
            if attr in self.read_only_attr_names:
                input = Label(self.init_window_name, text=value)
            else:
                if isinstance(value, bool):
                    if value is True:
                        combobox_idx = 0
                    else:
                        combobox_idx = 1
                    options_name = attr + "_option"
                    options = StringVar()
                    setattr(self, options_name, options)
                    input = ttk.Combobox(self.init_window_name, width=15, textvariable=options)
                    input['values'] = ("はい", "いいえ")
                    input.current(combobox_idx)
                else:
                    var = StringVar()
                    text_name = attr + "_text"
                    setattr(self, text_name, var)
                    input = Entry(self.init_window_name, textvariable=var, width=max(20, len(str(value))))
                    var.set(value)
                    print(value)

            setattr(self, input_name, input)
            input.grid(row=idx, column=1)

        self.save_button = Button(self.init_window_name, text="保存", bg="lightblue", width=10,
                                  command=self.save_config)
        self.save_button.grid(row=idx + 1, column=1)

    def save_config(self):
        for attr in self.config.__dict__:
            input_name = attr + "_text"
            input = getattr(self, input_name)
            if isinstance(input, Entry):
                new_value = input.get().strip()
                setattr(self.config, attr, new_value)
        import torch
        torch.save(self.config, self.config_save_path)

        # with open(self.config_save_path, 'wb') as f:
        #     pickle.dump(self.config, f)
        messagebox.showinfo("message", 'save successfully')

    def load_config(self):
        import torch
        self.config = torch.load(self.config_save_path)
        # with open(self.config_save_path, 'rb') as f:
        #     print(self.config_save_path)
        #     unpickler = pickle.Unpickler(f)
        #     self.config = unpickler.load()


def main():
    from configs.experiment_config import MnistConfig
    config = MnistConfig()
    init_window = Tk()
    ZMJ_PORTAL = ConfigGUI(init_window, config, GUI_config.experiment_config_path)
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()


if __name__ == '__main__':
    main()
