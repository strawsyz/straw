import wx

# 文本框输入test.txt
# 点击打开，就回打开test.txt文件

## 给按钮添加事件的方法
# loadButton.Bind(wx.EVT_BUTTON, load)


def load(event):
    file = open(filename.GetValue())  # 获得文件
    contents.SetValue(file.read())  # 读取文件内容
    file.close()

def save(event):
    file = open(filename.GetValue(),'w') #，打开文件
    file.write(contents.GetValue())# 写入文件
    file.close()

app = wx.App()
win = wx.Frame(None,title = "编辑器", size=(410,335))
bkg = wx.Panel(win)

loadButton = wx.Button(bkg, label = '打开')
loadButton.Bind(wx.EVT_BUTTON, load)  # 给按钮绑定读取文件的方法

saveButton = wx.Button(bkg, label = '保存')
saveButton.Bind(wx.EVT_BUTTON, save)  # 给按钮绑定写入文件的方法

filename = wx.TextCtrl(bkg)  # 显示文件名的格子
contents = wx.TextCtrl(bkg, style = wx.TE_MULTILINE | wx.HSCROLL)

# wx.BoxSizer的构造函数，默认是水平。wx.VERTICAL可以变成垂直
hbox = wx.BoxSizer()
# add方法中。
# proportion参数根据在窗口改变大小时所分配的空间设置比例。
# flag参数类似于构造函数中的style函数
hbox.Add(filename, proportion =1, flag = wx.EXPAND)
hbox.Add(loadButton, proportion =0,flag = wx.LEFT, border = 5)
hbox.Add(saveButton, proportion =0,flag = wx.LEFT, border = 5)

vbox = wx.BoxSizer(wx.VERTICAL)
vbox.Add(hbox,proportion = 0,flag = wx.EXPAND | wx.ALL, border = 5)
vbox.Add(contents, proportion = 1,flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border = 5)

bkg.SetSizer(vbox)
win.Show()
app.MainLoop()