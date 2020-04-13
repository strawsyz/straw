from django.contrib import admin

# Register your models here.
# 虽然这里报错，但是项目可以运行，要是改了的就不能运行了
from blog.models import BlogsPost
from blog.models import Admin
from blog.models import KeBiao
from blog.models import XueJi
from blog.models import BanJi
from blog.models import Illuster


# 用admin来管理数据
# Register your models here.
class BlogsPostAdmin(admin.ModelAdmin):
    list_display = ['title', 'body', 'timestamp']


admin.site.register(BlogsPost, BlogsPostAdmin)


class AdminAdmin(admin.ModelAdmin):
    list_display = ['uname', 'upass']


admin.site.register(Admin, AdminAdmin)


class KeBiaoAdmin(admin.ModelAdmin):
    list_display = ['kecheng', 'shijian', 'banji', 'jiaoshi']


admin.site.register(KeBiao, KeBiaoAdmin)


class XueJiAdmin(admin.ModelAdmin):
    list_display = ['xuehao', 'name', 'sex', 'chusheng', 'minzu', 'jiguan', 'zhuzi', 'phone']


admin.site.register(XueJi, XueJiAdmin)


class ClassAdmin(admin.ModelAdmin):
    list_display = ['cno', 'cname', 'ctno', 'cnumber']


admin.site.register(BanJi, ClassAdmin)


class IllusterAdmin(admin.ModelAdmin):
    list_display = ['illuster_id', 'name', 'image_url', 'modify_time']
    # reversion_enable = True


admin.site.register(Illuster, IllusterAdmin)
