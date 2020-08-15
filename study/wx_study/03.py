import wx

## 和exmaple2相同但是，使用了相对坐标，而不是绝对坐标


app = wx.App()
win = wx.Frame(None,title = "编辑器", size=(410,335))
bkg = wx.Panel(win)

loadButton = wx.Button(bkg, label = '打开')
saveButton = wx.Button(bkg, label = '保存')
filename = wx.TextCtrl(bkg)
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