from django.contrib.auth.models import Permission, User
from django.db import models


class Filter(models.Model):
    user = models.ForeignKey(User, default=1)
    artist = models.CharField(max_length=250)
    filter_title = models.CharField(max_length=250)
    filter_description = models.TextField(default='')
    filter_file = models.FileField(default='')

    def __str__(self):
        return self.artist + ': ' + self.filter_title


class Image(models.Model):
    user = models.ForeignKey(User, default=1)
    image_title = models.CharField(max_length=250)
    image_description = models.TextField(default='')
    image_file = models.FileField(default='')

    def __str__(self):
        return self.image_title


class Session(models.Model):
    user = models.ForeignKey(User, default=1)
    session_name = models.CharField(max_length=250)
    filter = models.ForeignKey(Filter, on_delete=models.CASCADE)
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    output_image_title = models.CharField(max_length=250)
    iterations = models.IntegerField(default=100)

    def __str__(self):
        return 'Session: ' + self.session_name + ' [Filter: ' + self.filter.filter_title + \
               ', Image: ' + self.image.image_title + ']'
