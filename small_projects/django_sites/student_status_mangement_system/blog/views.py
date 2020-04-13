from django import forms
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from blog.models import BlogsPost
from blog.models import Admin
from blog.models import XueJi
from blog.models import KeBiao
from blog.models import BanJi
from blog.models import Illuster
from blog.models import Illust

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Create your views here.
from django.views.decorators.csrf import csrf_protect


class UserForm(forms.Form):
    username = forms.CharField(label='用户名', max_length=100)
    password = forms.CharField(label='密__码', widget=forms.PasswordInput())


def blog_index(request):
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    # blog_list = BlogsPost.objects.all().values("title","body");
    blog = BlogsPost.objects.get(title='django学习');  # gei函数只能回传一个值 的时候才能用
    # import datetime
    # BlogsPost.objects.create(title=title, body=body, timestamp=datetime.datetime.now().date())
    # BlogsPost.objects.create(title="php学习", body="dasfhukwehfilwjefi的开奖号发卡机", timestamp=datetime.datetime.now().date())
    return render(request, 'blog_index.html', {'blog_list': blog_list})

    # return render(request,'blog_index.html', {'blog_list':blog_list})   # 返回blog_index.html


# blog添加测试
def blog_delete(request):
    BlogsPost.objects.filter(title=request.GET['d']).delete()
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    return render(request, 'blog_index.html', {'blog_list': blog_list})


# blog更新测试
def blog_update(request):
    # if request.method=="get":
    if request.POST:
        title = request.POST["title"]
        body = request.POST["body"]
        import datetime
        BlogsPost.objects.create(title=title, body=body, timestamp=datetime.datetime.now().date())
        blog_list = BlogsPost.objects.all()  # 获取所有数据
        return render(request, 'blog_index.html', {'blog_list': blog_list})
    blog = BlogsPost.objects.get(title=request.GET["u"])
    return render(request, "blog_update.html", {"blog": blog})


def blog_search(request):
    request.encoding = 'utf-8'
    if 'q' in request.GET:
        # message = '你搜索的内容为: ' + request.GET['q']+"<br/>"
        resilt = BlogsPost.objects.filter(title=request.GET['q'])  # filter韩慧能够传回多个结果
        # result = '你的搜索结果为：' + str(temp);
    else:
        message = '你提交了空表单'
    # return HttpResponse(message,result)
    return render(request, 'blog_index.html', {'blog_list': resilt})


@csrf_protect
def blogadd(request):
    blog_list = []
    if request.POST:
        title = request.POST["title"]
        body = request.POST["body"]
        import datetime
        BlogsPost.objects.create(title=title, body=body, timestamp=datetime.datetime.now().date())
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    return render(request, 'blog_index.html', {'blog_list': blog_list})


def test_login(request):
    return render(request, 'test.html', )


# 跳到登录页面
def admin_login(request):
    admin_list = Admin.objects.all()
    return render(request, 'login.html')


# 网页主页
def index(request):
    if request.POST:  # 如果是post上来的
        uname = request.POST["uname"]
        upass = request.POST["upass"]
        user = Admin.objects.get(uname=uname)
        # 对比提交的数据与数据库中的数据
        user = Admin.objects.filter(uname=uname, upass=upass)
        if user:
            request.session["uname"] = uname
            # response.set_cookie('uname', uname, 3600)
            return render(request, "index.html")
        else:
            error = "密码错误！"
            return render(request, 'login.html', {"error": error})
    return render(request, 'index.html')


def head(request):
    return render(request, 'head.html')


def left(request):
    return render(request, 'left.html')


def main(request):
    return render(request, 'main.html')


def tab(request):
    return render(request, 'tab.html')


# path('bC/',views.bC),
# path('bT/',views.bT),
# path('changepwd/',views.changepwd),
# path('head2/',views.head2),
# path('kC/',views.kC),
# path('kT/',views.kT),
# path('user/',views.user),
# path('useradd/',views.useradd),
# path('userupdate/',views.userupdate),
# path('xC/',views.xC),
# path('xT/',views.xT),
# path('yC/',views.yC),
# path('yT/',views.yT),
def bC(request):
    try:
        if request.GET["cno"]:
            # message = '你搜索的内容为: ' + request.GET['q']+"<br/>"
            resilt = BanJi.objects.filter(cno=request.GET['cno'])  # filter函数能够传回多个结果
            return render(request, 'bC.html', {"banji_list": resilt})
    except Exception:
        pass
    xueji_list = BanJi.objects.all()
    return render(request, 'bC.html', {"banji_list": xueji_list})


def bT(request):
    if request.POST:
        cno = request.POST["cno"]
        cname = request.POST["cname"]
        ctno = request.POST["ctno"]
        cnumber = request.POST["cnumber"]
        try:
            banji = BanJi.objects.get(cno=cno)
        except Exception:
            BanJi.objects.create(cno=cno, cname=cname, ctno=ctno, cnumber=cnumber)
            response = HttpResponseRedirect('/left/bC/')
            return response
        banji.cname = cname
        banji.ctno = ctno
        banji.cnumber = cnumber
        banji.save()
        response = HttpResponseRedirect('/left/bC/')
        return response
    try:
        if request.GET["u"]:  # 说明是跳转到跟新页面的
            banji = BanJi.objects.get(cno=request.GET["u"])
            return render(request, 'bT.html', {"banji": banji})
    except Exception as e:
        try:
            if request.GET["d"]:
                BanJi.objects.filter(cno=request.GET['d']).delete()
                response = HttpResponseRedirect('/left/bC/')
                return response
        except Exception as e:
            print(e)
    return render(request, 'bT.html')


def changepwd(request):
    return render(request, 'changepwd.html')


def head2(request):
    # 删除 session
    response = HttpResponseRedirect('/login/')
    del request.session["uname"]  # 不存在时报错
    return response
    # return render(request,'login.html')


# 课表查询
def kC(request):
    try:
        if request.GET["banji"]:
            # message = '你搜索的内容为: ' + request.GET['q']+"<br/>"
            resilt = KeBiao.objects.filter(banji=request.GET['banji'])  # filter函数能够传回多个结果
            if request.GET["kecheng"]:
                resilt = KeBiao.objects.filter(banji=request.GET['banji'], kecheng=request.GET['kecheng'])
            return render(request, 'kC.html', {"kebiao_list": resilt})

        elif request.GET["kecheng"]:
            resilt = KeBiao.objects.filter(kecheng=request.GET['kecheng'])
            return render(request, 'kC.html', {"kebiao_list": resilt})
    except Exception:
        pass
    kebiao_list = KeBiao.objects.all()
    return render(request, 'kC.html', {"kebiao_list": kebiao_list})


# 课表添加h，还有一些其他操作
def kT(request):
    if request.POST:
        kecheng = request.POST["kecheng"]
        shijian = request.POST["shijian"]
        banji = request.POST["banji"]
        jiaoshi = request.POST["jiaoshi"]
        try:
            kebiao = KeBiao.objects.get(kecheng=kecheng)
        except Exception:
            KeBiao.objects.create(kecheng=kecheng, shijian=shijian, banji=banji, jiaoshi=jiaoshi)
            response = HttpResponseRedirect('/left/kC/')
            return response
        kebiao.banji = banji
        kebiao.shijian = shijian
        kebiao.jiaoshi = jiaoshi
        kebiao.save()
        # blog_list = KeBiao.objects.all()  # 获取所有数据
        response = HttpResponseRedirect('/left/kC/')
        return response
    try:
        if request.GET["u"]:  # 说明是跳转到跟新页面的
            kebiao = KeBiao.objects.get(kecheng=request.GET["u"])
            return render(request, 'kT.html', {"kebiao": kebiao})
    except Exception as e:
        try:
            if request.GET["d"]:
                KeBiao.objects.filter(kecheng=request.GET['d']).delete()
                response = HttpResponseRedirect('/left/kC/')
                return response
        except Exception as e:
            print(e)
    return render(request, 'kT.html')


def user(request):
    return render(request, 'user.html')


def useradd(request):
    return render(request, 'useradd.html')


def userupdate(request):
    return render(request, 'userupdate.html')


def xC(request):
    try:
        if request.GET["name"]:
            # message = '你搜索的内容为: ' + request.GET['q']+"<br/>"
            resilt = XueJi.objects.filter(name=request.GET['name'])  # filter函数能够传回多个结果
            if request.GET["xuehao"]:
                resilt = XueJi.objects.filter(name=request.GET['name'], xuehao=request.GET['xuehao'])
            return render(request, 'xC.html', {"xueji_list": resilt})

        elif request.GET["xuehao"]:
            resilt = XueJi.objects.filter(xuehao=request.GET['xuehao'])
            return render(request, 'xC.html', {"xueji_list": resilt})
    except Exception:
        pass
    xueji_list = XueJi.objects.all()
    return render(request, 'xC.html', {"xueji_list": xueji_list})


def xT(request):
    if request.POST:
        xuehao = request.POST["xuehao"]
        name = request.POST["name"]
        sex = request.POST["sex"]
        chusheng = request.POST["chusheng"]
        minzu = request.POST["minzu"]
        jiguan = request.POST["jiguan"]
        zhuzi = request.POST["zhuzi"]
        phone = request.POST["phone"]
        try:
            xueji = XueJi.objects.get(xuehao=xuehao)
        except Exception:
            XueJi.objects.create(xuehao=xuehao, name=name, sex=sex, chusheng=chusheng, minzu=minzu, jiguan=jiguan,
                                 zhuzi=zhuzi, phone=phone)
            response = HttpResponseRedirect('/left/xC/')
            return response
        xueji.name = name
        xueji.sex = sex
        xueji.chusheng = chusheng
        xueji.minzu = minzu
        xueji.jiguan = jiguan
        xueji.chusheng = chusheng
        xueji.chusheng = chusheng
        xueji.chusheng = chusheng
        xueji.save()
        response = HttpResponseRedirect('/left/xC/')
        return response
    try:
        if request.GET["u"]:  # 说明是跳转到跟新页面的
            xueji = XueJi.objects.get(xuehao=request.GET["u"])
            return render(request, 'xT.html', {"xueji": xueji})
    except Exception as e:
        try:
            if request.GET["d"]:
                XueJi.objects.filter(xuehao=request.GET['d']).delete()
                response = HttpResponseRedirect('/left/xC/')
                return response
        except Exception as e:
            print(e)
    return render(request, 'xT.html')


def yT(request):
    if request.POST:
        sno = request.POST["sno"]
        sname = request.POST["sname"]
        ssex = request.POST["ssex"]
        sclass = request.POST["sclass"]
        try:
            yonghu = Illuster.objects.get(sno=sno)
        except Exception:
            Illuster.objects.create(sno=sno, sname=sname, ssex=ssex, sclass=sclass)
            response = HttpResponseRedirect('/left/yC/')
            return response
        yonghu.sname = sname
        yonghu.ssex = ssex
        yonghu.sclass = sclass
        yonghu.save()
        response = HttpResponseRedirect('/left/yC/')
        return response
    try:
        if request.GET["u"]:  # 说明是跳转到跟新页面的
            yonghu = Illuster.objects.get(sno=request.GET["u"])
            return render(request, 'yT.html', {"yonghu": yonghu})
    except Exception as e:
        try:
            if request.GET["d"]:
                Illuster.objects.filter(sno=request.GET['d']).delete()
                response = HttpResponseRedirect('/left/yC/')
                return response
        except Exception as e:
            print(e)
    return render(request, 'yT.html')


def illuster_list(request):
    # try:
    illuster_list = None
    if "name" in request.GET:
        illuster_list = Illuster.objects.filter(name=request.GET['name'])
    if "illuster_id" in request.GET:
        illuster_list = Illuster.objects.filter(illuster_id=request.GET['illuster_id'])
    # except Exception:
    #     pass
    if illuster_list is None:
        illuster_list = Illuster.objects.filter(priority=0).values('illuster_id', 'name',
                                                                   'image_url',
                                                                   'modify_time').order_by(
            "-illuster_id")
    paginator = Paginator(illuster_list, 25)
    page = request.GET.get('page')
    try:
        illusters = paginator.page(page)
    except PageNotAnInteger:
        illusters = paginator.page(1)
    except EmptyPage:
        illusters = paginator.page(paginator.num_pages)
    return render(request, 'illuster_list.html', {"illuster_list": illusters})


def illuster_info(request):
    if "id" in request.GET:
        illust_list = Illust.objects.filter(illuster_id=request.GET['id']).values('illust_id', 'title',
                                                                                  'page_no',
                                                                                  'loc_url').order_by(
            "-illust_id")
        paginator = Paginator(illust_list, 25)
        page = request.GET.get('page')
        try:
            illust_list = paginator.page(page)
        except PageNotAnInteger:
            illust_list = paginator.page(1)
        except EmptyPage:
            illust_list = paginator.page(paginator.num_pages)
        return render(request, 'illuster_info.html', {"illust_list": illust_list, "illuster_id": request.GET['id']})
    else:
        return render(request, 'error.html')


def illust_info(request):
    if "id" in request.GET:
        illust_info = Illust.objects.filter(illust_id=request.GET['id']).values('loc_url', 'title', 'page_no',
                                                                                'illuster_id')
        if illust_info is None:
            return render(request, 'error.html')
        illust_info = list(illust_info)[0]
        illuster_id = illust_info['illuster_id']
        title = illust_info['title']
        loc_url = illust_info['loc_url']
        illust_list = []
        for i in range(illust_info['page_no']):
            illust_list.append(loc_url.replace('_p0', '_p' + str(i)))

        return render(request, 'illust_info.html',
                      {"title": title, "illust_list": illust_list, 'illuster_id': illuster_id})
        # if request.GET["illuster_id"]:
        #     illuster_list = Illust.objects.filter(sname=request.GET['name'],sno=request.GET['illuster_id'])
        # return render(request, 'illuster_list.html', {"illuster_list": resilt})
    else:
        return render(request, 'error.html')
