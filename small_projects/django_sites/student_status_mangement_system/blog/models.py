from django.db import models


# Create your models here.
# 定义Blog的数据结构
class BlogsPost(models.Model):
    title = models.CharField(max_length=150)  # 博客标题
    body = models.TextField()  # 博客正文
    timestamp = models.DateTimeField()  # 创建时间


class KeBiao(models.Model):
    kecheng = models.CharField(max_length=20)
    shijian = models.CharField(max_length=6)
    banji = models.CharField(max_length=10)
    jiaoshi = models.CharField(max_length=10)


class XueJi(models.Model):
    xuehao = models.CharField(max_length=20)
    name = models.CharField(max_length=20)
    sex = models.CharField(max_length=3)
    chusheng = models.CharField(max_length=20)
    minzu = models.CharField(max_length=5)
    jiguan = models.CharField(max_length=20)
    zhuzi = models.CharField(max_length=30)
    phone = models.CharField(max_length=15)


class Admin(models.Model):
    uname = models.CharField(max_length=20)  # 登录用的
    upass = models.CharField(max_length=20)  # 登录用的


class Illuster(models.Model):
    illuster_id = models.CharField(max_length=20)  # 学号
    name = models.CharField(max_length=10)  # 学生姓名
    image_url = models.CharField(max_length=10)  # 所在班级
    modify_time = models.DateTimeField()  # 性别
    priority = models.BigIntegerField()  # 性别

    class Meta:
        db_table = 'illuster'  # 更改django默认前缀数据库命名规则
        ordering = ['modify_time']


class BanJi(models.Model):
    cno = models.CharField(max_length=10)  # 班级名
    cname = models.CharField(max_length=10)  # 班主任姓名
    ctno = models.CharField(max_length=20)  # 班主任工号
    cnumber = models.CharField(max_length=20)  # 班主任联系方式


class Illust(models.Model):
    illust_id = models.CharField(max_length=20)  # 学号
    title = models.CharField(max_length=10)  # 学生姓名
    page_no = models.CharField(max_length=10)  # 所在班级
    url = models.CharField(max_length=100)  # 性别
    illuster_id = models.CharField(max_length=20)  # 学号
    loc_url = models.CharField(max_length=50)
    class Meta:
        db_table = 'illust'  # 更改django默认前缀数据库命名规则
        ordering = ['illust_id']
