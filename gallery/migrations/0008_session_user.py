# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-05-23 04:27
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('gallery', '0007_auto_20170523_1324'),
    ]

    operations = [
        migrations.AddField(
            model_name='session',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]