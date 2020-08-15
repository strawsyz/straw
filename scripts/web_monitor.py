import urllib.request
import time
import os

web_url = "/"
html_save_path = ''
while True:
    print("start!!!!!!")
    if not os.path.exists(html_save_path):
        size = 0
    else:
        size = os.stat(html_save_path)[6]
    time.sleep(4)  # 每隔一天运行一次 24*60*60=86400s
    file = urllib.request.urlopen(web_url)
    data = file.read()  # 读取页面放到data变量中
    ffile = open(html_save_path, "wb+")
    ffile.write(data)  # 把页面内容写到文件中
    ffile.close()
    newsize = os.stat(html_save_path)[6]
    if size != newsize:
        break
# todo
print("更新了！！！!!!")
