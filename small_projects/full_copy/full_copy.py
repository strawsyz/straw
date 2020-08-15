import re
import pyperclip
from PyQt5.QtWidgets import *

'''可以监控粘贴板，
还需要一个图形界面'''

app = QApplication([])
clipboard = QApplication.clipboard()
limit_length = 100
limit_num_of_words_in_history = 10
magent_pattern = re.compile(r'[0-9a-zA-Z]{40}')
history = []
current = ''


def add_into_history(num_words, history):
    """
    添加新的复制的文字到历史列表中
    :return:
    """
    if len(history) > limit_num_of_words_in_history:
        history = history[1:]
        history.append(num_words)
    return history


def get_clip_text():
    """
    读取剪切板的文字
    :return:
    """
    temp = pyperclip.paste().strip()
    global current
    print(12)
    if temp != '' and temp != current and len(temp) < limit_length:

        # 判断是否有超链接
        res = re.findall(magent_pattern, temp)
        if len(res) == 1:
            history.append(temp)
            temp = 'magnet:?xt=urn:btih:' + res[0]
            pyperclip.copy(temp)
        current = temp
        history.append(temp)
        print(history)
        print(current)

        # if temp == 'end':
        #     break
        #     time.sleep(1)


clipboard.dataChanged.connect(get_clip_text)

app.exec_()
