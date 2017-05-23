from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'gallery'

urlpatterns = [
    url(r'^$', views.UserFormView.as_view(), name='visitor'),
    url(r'^login_user/$', views.login_user, name='login_user'),
    url(r'^logout_user/$', views.logout_user, name='logout_user'),
    url(r'user/$', views.IndexView.as_view(), name='user'),
    url(r'^filters/$', views.filters, name='filters'),
    url(r'^images/$', views.images, name='images'),
    url(r'^convert/$', views.convert, name='convert'),
    url(r'^create_image/$', views.create_image, name='create_image'),
    url(r'^(?P<image_id>[0-9]+)/delete_image/$', views.delete_image, name='delete_image'),
    url(r'^create_filter/$', views.create_filter, name='create_filter'),
    url(r'^(?P<filter_id>[0-9]+)/delete_filter/$', views.delete_filter, name='delete_filter'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)