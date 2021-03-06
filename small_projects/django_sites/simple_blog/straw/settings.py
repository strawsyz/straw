"""
Django settings for gushao project.

Generated by 'django-admin startproject' using Django 2.0.5.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'c+dql!wwca64nn_4nu+5+x3h5*st_5lece_mxp@d3f3f#tj)nk'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']
#配置静态文件
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "static"),
)


# Application definition

INSTALLED_APPS = [
    'suit',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'straw.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',  # <-更改admin后台,需要这一行
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
# DATETIME_FORMAT = 'Y-m-d H:i:s'
# DATE_FORMAT = 'Y-m-d'
# LANGUAGE_CODE = 'zh_CN'
# TIME_ZONE = 'Asia/Shanghai'
# USE_I18N = True
# USE_L10N = False
# USE_TZ = True
LANGUAGE_CODE = 'zh-Hans'

TIME_ZONE = 'Asia/Shanghai'
# LANGUAGE_CODE = 'zh_Hans'  # 设置成中文，老版本django使用'zh_CN'
# TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_L10N = False  # 注意是False 配合下边时间格式
USE_TZ = False  # 如果只是内部使用的系统，这行建议为false，不然会有时区问题
DATETIME_FORMAT = 'Y-m-d H:i:s'  # suit在admin里设置时间的一个小bug。需要把时间格式指定一下
DATE_FORMAT = 'Y-m-d'

SUIT_CONFIG = {
    'ADMIN_NAME': '在线用户：顾少', # 登录界面提示
    'LIST_PER_PAGE': 5,
    # 'MENU': ({'label': u'用户管理', 'app': 'blog', 'models': ('blog.UserProfile', 'blog.BlogsPost', 'blog.Short')})
             # 每一个字典表示左侧菜单的一栏
    'MENU': ({'label': '博客',  'icon': 'icon-user',   'app':'blog',  'models': ({'label':'用户','model':'blog.UserProfile'})},
                  # {'label': '博客文章', 'app': 'blog','model':({'label':'博客','model':'blog.BlogsPost'})},
             # {'label': '碎碎念', 'app': 'blog', 'model': ('blog.Short')}
             ),
    # 每一个字典表示左侧菜单的一栏
    # label表示name，app表示上边的install的app，models表示用了哪些models
}

WSGI_APPLICATION = 'straw.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

# LANGUAGE_CODE = 'en-us'
#
# TIME_ZONE = 'UTC'
#
# USE_I18N = True
#
# USE_L10N = True
#
# USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'

TEMPLATE_DIRS = (os.path.join(BASE_DIR,  'templates'),)
