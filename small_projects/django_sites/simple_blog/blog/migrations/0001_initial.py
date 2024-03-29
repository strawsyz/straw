# Generated by Django 2.0.5 on 2018-06-21 09:02

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BlogsPost',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=20)),
                ('img', models.CharField(max_length=50)),
                ('body', models.TextField()),
                ('timestamp', models.DateTimeField()),
                ('author', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='Short',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('timestamp', models.DateTimeField()),
                ('content', models.CharField(max_length=100)),
            ],
        ),
    ]
