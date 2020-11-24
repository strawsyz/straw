"""HelloWorld URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from blog import views

urlpatterns = [
    # 学籍管理部分
    path('admin/', admin.site.urls),
    path('blog/', views.blog_index),
    path('blogadd/', views.blogadd),
    path('test/', views.test_login),
    path('login/', views.admin_login),
    path('blog_search/', views.blog_search),
    path('blog_delete/', views.blog_delete),
    path('blog_update/', views.blog_update),
    path('head/', views.head),
    path('left/', views.left),
    path('main/', views.image_checker),
    path('tab/', views.tab),
    path('left/bC/', views.bC),
    path('left/bT/', views.bT),
    path('changepwd/', views.changepwd),
    path('head/head2/', views.head2),
    path('left/kC/', views.kC),
    path('left/kT/', views.kT),
    path('left/user/', views.user),
    path('useradd/', views.useradd),
    path('userupdate/', views.userupdate),
    path('left/xC/', views.xC),
    path('left/xT/', views.xT),
    path('left/yT/', views.yT),
    # 插画网站部分
    url('left/illuster_list/', views.illuster_list, name="illuster_list"),
    url('illuster_info/', views.illuster_info, name="illuster_info"),
    url('illust_info/', views.illust_info, name="illust_info"),

    path('', views.index)
]
