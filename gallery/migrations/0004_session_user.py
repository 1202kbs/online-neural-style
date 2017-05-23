# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-05-22 13:51
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('gallery', '0003_auto_20170522_1428'),
    ]

    operations = [
        migrations.AddField(
            model_name='session',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]