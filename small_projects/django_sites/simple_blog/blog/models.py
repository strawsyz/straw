from django.db import models
from django.contrib.auth.models import User


# Create your models here.
# 定义Blog的数据结构
class UserProfile(User):
    cname = models.CharField("中文名称", max_length=30)


class BlogsPost(models.Model):
    id = models.IntegerField(primary_key=True)  # 自增主键
    title = models.CharField(max_length=100)  # 博客标题
    type = models.CharField(max_length=20)  # 博客类型
    img = models.CharField(max_length=50)  # 图片路径
    body = models.TextField()  # 博客正文
    timestamp = models.DateTimeField()  # 创建时间
    author = models.CharField(max_length=10)  # 创建人


#
class Short(models.Model):
    id = models.IntegerField(primary_key=True)
    timestamp = models.DateTimeField()  # 创建时间
    content = models.CharField(max_length=100)  # 内容
