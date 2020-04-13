from django.contrib import admin
from blog.models import BlogsPost
from blog.models import Short

# # Register your models here.
# # Register your models here.
from django.contrib import admin
from blog import models


class UserProfileAdmin(admin.ModelAdmin):
    # 定义admin总览里每行的显示信息
    list_display = ('cname', 'username', 'email')
    # 定义搜索框以哪些字段可以搜索
    search_fields = ('cname', 'username')


# 引用的固定格式，注册的model和对应的Admin，Admin放在后边
# 同样还有noregister方法：比如admin.site.noregister(Group)，把group这个表在admin中去掉（默认user和group都是注册到admin中的）
admin.site.register(models.UserProfile, UserProfileAdmin)


class BlogsPostAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'type', 'img', 'body', 'timestamp', 'author']


admin.site.register(BlogsPost, BlogsPostAdmin)


class ShortAdmin(admin.ModelAdmin):
    list_display = ['id', 'timestamp', 'content']


admin.site.register(Short, ShortAdmin)
