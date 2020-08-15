import wx
# 只是有个能看的界面
app = wx.App()
win = wx.Frame(None,title = "编辑器", size=(410,335))
win.Show()

loadButton = wx.Button(win, label = '打开',pos = (225,5),size = (80,25))
saveButton = wx.Button(win, label = '保存',pos = (315,5),size = (80,25))
filename = wx.TextCtrl(win, pos = (5,5),size = (210,25))
# wx.TE_MULTILINE 能显示多行
# wx.HSCROLL 水平滚动条
# or 可以换成 | 符号
contents = wx.TextCtrl(win, pos = (5,35),size = (390,260), style = wx.TE_MULTILINE | wx.HSCROLL)
app.MainLoop()