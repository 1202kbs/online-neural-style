# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-05-22 05:24
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('gallery', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Session',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_name', models.CharField(max_length=250)),
                ('output_image_title', models.CharField(max_length=250)),
            ],
        ),
        migrations.AddField(
            model_name='filter',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='image',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='session',
            name='filter',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='gallery.Filter'),
        ),
        migrations.AddField(
            model_name='session',
            name='image',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='gallery.Image'),
        ),
    ]
