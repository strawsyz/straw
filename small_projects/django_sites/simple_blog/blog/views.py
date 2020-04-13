from django.shortcuts import render
from blog.models import BlogsPost
from blog.models import Short


# Create your views here.
def index(request):
    blog_list = BlogsPost.objects.all()[:6]
    for i in blog_list:
        i.body = i.body[:100]
        print(i.body)
        print(i)
    # 最新文章
    new_blogs = BlogsPost.objects.all()[:8]
    # 排行
    blogs = blog_list[:5]
    return render(request, 'index.html', {'blog_list': blog_list, 'new_blogs': new_blogs, 'blogs': blogs})


def about(request):
    return render(request, 'about.html')


def moodlist(request):
    mood_list = Short.objects.all()  # 获取所有数据
    # for i in mood_list:
    #     i.timestamp.fromtimestamp()
    return render(request, 'moodlist.html', {'moood_list': mood_list})


def newlist(request):
    blog_list = BlogsPost.objects.filter(type="慢生活")
    print(blog_list)
    for i in blog_list:
        i.body = i.body[:100]
    return render(request, 'newlist.html', {'blog_list': blog_list})


def knowledge(request):
    blog_list = BlogsPost.objects.filter(type="学无止境")
    print(blog_list)
    for i in blog_list:
        i.body = i.body[:100]
    return render(request, 'knowledge.html', {'blog_list': blog_list})


def new(request):
    blog = BlogsPost.objects.get(id=request.GET['id'])  # filter能够传回多个结果
    return render(request, 'new.html', {'blog': blog})
