import sys
from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui
import webbrowser
import subprocess
from conf_util import ConfigureUtil

class WebPage(QWebPage):

    def __init__(self):
        super(WebPage, self).__init__()

    def acceptNavigationRequest(self, frame, request, type):
        # 接受请求的方法
        if (type == QWebPage.NavigationTypeLinkClicked):
            if (frame == self.mainFrame()):
                self.view().load(request.url())
                self.addressBar.setText(request.url().toString())
                # print("local window")
            else:
                # 部分的超链接，会出现在这里
                # webbrowser.open(request.url().toString())
                self.view().load(request.url())
                # self.addressBar.setText(request.url().toString())
                # QWebView().load(request.url()).show()
                # web.show()
                # print("test")
                return False
        return QWebPage.acceptNavigationRequest(self, frame, request, type)


class MyBrowser(QWidget):
    def __init__(self, parent=None):
        # 加载配置文件
        self.load_config()

        super(MyBrowser, self).__init__(parent)
        #设置图标
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(self.APP_ICON), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        # palette = QPalette()
        # palette.setColorGroup()
        # button=palette.button()

        QPalette
        pal = QPalette();
        pal.setColor(QPalette.ButtonText, Qt.red)
        # // pal.setColor(QPalette::Button, Qt::green);
        self.setPalette(pal);
        # self.setStyleSheet("background-color:green");
        self.setPalette(QPalette(QColor(250, 250, 200)));
        # self.setWindowOpacity(0.5)#设置透明度
        self.createLayout()#布置界面
        self.createConnection()#给按钮关联方法
        self.webView.load(QUrl(self.MAIN_PAGE))
        self.webView.show()
        #加代理
        # proxy = QNetworkProxy()
        # proxy.setType(QNetworkProxy.HttpProxy)
        # proxy.setHostName("127.0.0.1")
        # proxy.setPort(9666)
        # QNetworkProxy.setApplicationProxy(proxy)

        self.setGeometry(self.APP_LEFT, self.APP_TOP, self.APP_WIDTH, self.APP_HEIGHT)

    def load_config(self):
        self.CONFIG = ConfigureUtil()
        self.MAIN_PAGE = self.CONFIG.get('mainpage', 'url')
        self.PROXY_PATH = self.CONFIG.get('proxy', 'path')
        self.PROXY_HOST = self.CONFIG.get('proxy', 'host')
        self.PROXY_PORT = self.CONFIG.get('proxy', 'port', 'int')
        self.FAVORITE_PAGES = self.CONFIG.get_section('pages')
        self.PI_PAGES = self.CONFIG.get_section('pi_pages')
        self.PI_URL_PRE = self.CONFIG.get('pi', 'urlPre')
        self.APP_ICON = self.CONFIG.get('app', 'icon')
        self.APP_TOP = self.CONFIG.get('app', 'top', 'int')
        self.APP_LEFT = self.CONFIG.get('app', 'left', 'int')
        self.APP_HEIGHT = self.CONFIG.get('app', 'height', 'int')
        self.APP_WIDTH = self.CONFIG.get('app', 'width', 'int')

    def search(self):
        address = str(self.addressBar.text())
        if address:
            url = QUrl(address)
            self.webView.load(url)
            self.webView.show()

    def proxy(self):
        # subprocess.Popen(self.PROXY_PATH)  # 打开代理软件
        proxy = QNetworkProxy()
        proxy.setType(QNetworkProxy.HttpProxy)
        proxy.setHostName(self.PROXY_HOST)
        proxy.setPort(self.PROXY_PORT)
        QNetworkProxy.setApplicationProxy(proxy)
        self.sendButton.show()
        self.goButton.hide()
        print("use proxy")

    def disProxy(self):
        # 设置不使用代理
        proxy = QNetworkProxy()
        proxy.setType(QNetworkProxy.NoProxy)
        QNetworkProxy.setApplicationProxy(proxy)
        self.goButton.show()
        self.sendButton.hide()
        print("stop using proxy")


    def linkClicked(self, url):
        self.webView.load(url)
        self.addressBar.setText(url.toString())

    def createButton(self, text):
        button = QPushButton(text)
        color = QColor()
        color.setRgb(216, 223, 223)
        button.setPalette(QPalette(color))  # 鼠标放上去时的颜色
        button.setStyleSheet("color:red")
        return button

    def createLayout(self):
        self.setWindowTitle("S_browser")

        self.addressBar = QLineEdit()
        self.reloadButton = QPushButton("refresh")
        self.reloadButton.setStyleSheet("background-color:red")
        self.backButton = QPushButton("back")
        self.stopButton = QPushButton("stop")
        self.forwardButton = QPushButton("next")
        self.goButton = QPushButton("VPN")
        self.sendButton = QPushButton(" stop VPN")
        self.openButton = QPushButton("open in browser")
        self.toumingButton =QPushButton("transparent")
        self.butouButton = QPushButton("untransparent")
        self.backButton.setStyleSheet("background-color:red")
        self.backButton.setShortcut('Ctrl+Z')
        self.stopButton.setStyleSheet("background-color:red")
        self.forwardButton.setStyleSheet("background-color:red")
        self.goButton.setStyleSheet("background-color:red")
        self.openButton.setStyleSheet("background-color:red")
        self.toumingButton.setStyleSheet("background-color:red")
        # self.butouButton.setStyleSheet("background-color:red")

        truebutton = QPushButton()
        truebutton.setPalette(QPalette(QColor(222, 222, 111)))

        b2 = QHBoxLayout()  # 收藏夹1
        #  添加收藏夹部分
        for i in self.FAVORITE_PAGES:
            temp = self.createButton(i[0])
            b2.addWidget(temp)
            # 给按钮设置颜色
            # temp.setPalette(QPalette(QColor(111,111,111)))
            self.connect(temp, SIGNAL('clicked()'), self.jump(i[1]))

        b3 = QHBoxLayout()  # 收藏夹1
        #  添加收藏夹部分
        for i in self.PI_PAGES:
            temp = self.createButton(i[0])
            b3.addWidget(temp)
            # 给按钮设置颜色
            temp.setPalette(QPalette(QColor(111,111,111)))
            self.connect(temp, SIGNAL('clicked()'), self.jump(self.PI_URL_PRE+i[1]))


        bl = QHBoxLayout()

        bl.addWidget(self.reloadButton)
        bl.addWidget(self.backButton)
        bl.addWidget(self.stopButton)
        bl.addWidget(self.forwardButton)
        bl.addWidget(self.addressBar)
        bl.addWidget(self.goButton)
        bl.addWidget(self.sendButton)
        self.sendButton.hide()
        bl.addWidget(self.openButton)
        bl.addWidget(self.toumingButton)
        self.butouButton.hide()
        bl.addWidget(self.butouButton)

        self.webView = QWebView()
        self.webView.setPage(WebPage())

        self.webSettings = self.webView.settings()
        self.webSettings.setAttribute(QWebSettings.PluginsEnabled, True)
        self.webSettings.setAttribute(QWebSettings.JavascriptEnabled, True)
        self.webSettings.setAttribute(QWebSettings.JavascriptCanAccessClipboard, True)
        self.webSettings.setAttribute(QWebSettings.JavascriptCanCloseWindows, True)
        self.webSettings.setAttribute(QWebSettings.JavascriptCanOpenWindows, True)
        self.webSettings.setAttribute(QWebSettings.LocalStorageDatabaseEnabled, True)
        self.webSettings.setAttribute(QWebSettings.LocalStorageEnabled, True)
        self.webSettings.setAttribute(QWebSettings.OfflineWebApplicationCacheEnabled, True)
        self.webSettings.setAttribute(QWebSettings.OfflineStorageDatabaseEnabled, True)

        self.webView.page().setLinkDelegationPolicy(QWebPage.DelegateAllLinks)
        self.webView.page().linkClicked.connect(self.linkClicked)

        layout = QVBoxLayout()
        layout.addLayout(bl)
        layout.addLayout(b2)
        layout.addLayout(b3)

        layout.addWidget(self.webView)
        # 下面设置大小之后会无法改变大小
        # self.setFixedWidth(1300)
        # self.setFixedHeight(768)
        self.setLayout(layout)

    def jump(self, url):
        def jump():
            self.webView.load(QUrl(url))
            self.webView.show()
        return jump

    def stop_page(self):
        """
        Stop loading the page
        """
        self.webView.stop()

    def reload_page(self):
        """
        Reload the web page
        """
        self.webView.reload()
        # self.webView.setUrl(QtCore.QUrl(self.webView.url().toString()))

    def back(self):
        """
        Back button clicked, go one page back
        """
        self.webView.back()
        # page = self.ui.webView.page()
        # history = page.history()
        # history.back()
        # if self.webView.canGoBack():
        #     self.backButton.setEnabled(True)
        # else:
        #     self.backButton.setEnabled(False)

    def next(self):
        """
        Next button clicked, go to next page
        """
        # page = self.ui.webView.page()
        # history = page.history()
        # history.forward()
        self.webView.forward()
        # if self.webView.canGoForward():
        #     self.forwardButton.setEnabled(True)
        # else:
        #     self.forwardButton.setEnabled(False)

    def openBrowser(self):
        webbrowser.open_new_tab(self.addressBar.text())

    def urlChange(self):
        self.addressBar.setText(self.webView.url().toString())

    def touming(self):
        self.setWindowOpacity(0.7)
        self.butouButton.show()
        self.toumingButton.hide()

    def butou(self):
        self.setWindowOpacity(1)
        self.toumingButton.show()
        self.butouButton.hide()

    def createConnection(self):
        self.connect(self.addressBar, SIGNAL('returnPressed()'), self.search)
        self.connect(self.addressBar, SIGNAL('returnPressed()'), self.addressBar, SLOT('selectAll()'))
        self.connect(self.goButton, SIGNAL('clicked()'), self.proxy)
        self.connect(self.sendButton, SIGNAL('clicked()'), self.disProxy)
        self.connect(self.backButton, QtCore.SIGNAL("clicked()"), self.back)
        self.connect(self.reloadButton, QtCore.SIGNAL("clicked()"), self.reload_page)
        self.connect(self.forwardButton, QtCore.SIGNAL("clicked()"), self.next)
        self.connect(self.stopButton, QtCore.SIGNAL("clicked()"), self.stop_page)
        self.connect(self.openButton, QtCore.SIGNAL("clicked()"), self.openBrowser)
        self.connect(self.toumingButton ,QtCore.SIGNAL("clicked()"), self.touming)
        self.connect(self.butouButton ,QtCore.SIGNAL("clicked()"), self.butou)
        # 跳转页面之后，自动修改地址栏
        QtCore.QObject.connect(self.webView, QtCore.SIGNAL("urlChanged (const QUrl&)"), self.urlChange)
        # QWebView().urlChanged()
        
    # 解决窗口关闭报错的问题
    def closeEvent(self, event):
        quit()



app = QApplication(sys.argv)

browser = MyBrowser()
browser.show()
sys.exit(app.exec_())