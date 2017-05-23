from django.contrib.auth.models import User
from django import forms
from .models import Image, Filter, Session


class FilterForm(forms.ModelForm):

    class Meta:
        model = Filter
        fields = ['artist', 'filter_title', 'filter_description', 'filter_file']


class ImageForm(forms.ModelForm):

    class Meta:
        model = Image
        fields = ['image_title', 'image_description', 'image_file']


class SessionForm(forms.ModelForm):

    class Meta:
        model = Session
        fields = ['session_name', 'filter', 'image', 'output_image_title', 'iterations']


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
