from django.contrib.auth import authenticate, login, logout
from django.views import generic
from django.db.models import Q
from django.views.generic import View
from django.shortcuts import render, redirect, get_object_or_404
from .models import Image, Filter, Session
from .forms import UserForm, ImageForm, FilterForm, SessionForm
from . import neural_style as ns


# The view returned for a user
class IndexView(generic.ListView):
    template_name = 'convert/user.html'
    context_object_name = 'filters'

    def get_queryset(self):
        return Filter.objects.all()


def filters(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        filters = Filter.objects.filter(user=request.user)
        query = request.GET.get('q')
        if query:
            filters = filters.filter(
                Q(filter_title__icontains=query) |
                Q(artist__icontains=query)
            ).distinct()
            return render(request, 'convert/filters.html', {'filters': filters})
        else:
            return render(request, 'convert/filters.html', {'filters': filters})

        # return render(request, 'convert/filters.html', {'filters': filters})


def images(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        images = Image.objects.filter(user=request.user)
        query = request.GET.get('q')
        if query:
            images = images.filter(
                Q(image_title__icontains=query)
            ).distinct()
            return render(request, 'convert/images.html', {'images': images})
        else:
            return render(request, 'convert/images.html', {'images': images})


def convert(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        form = SessionForm(request.POST or None, request.FILES)

        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user

            filter_id = request.POST.get('filter')
            image_id = request.POST.get('image')
            output_title = request.POST.get('output_image_title')
            output_image_path = 'gallery/static/convert/output/' + output_title
            iterations = request.POST.get('iterations')

            filter = Filter.objects.get(pk=int(filter_id))
            image = Image.objects.get(pk=int(image_id))

            parser = ns.build_parser()
            options = parser.parse_args(['gallery/' + image.image_file.url[1:], 'gallery/' + filter.filter_file.url[1:], output_image_path, str(iterations)])
            ns.main(options)

            session.save()

            session_id = session.id
            session_input = Session.objects.get(pk=session_id)
            return render(request, 'convert/download.html', {'session': session_input})

        context = {
            'form': form,
        }

        return render(request, 'convert/sessions.html', context)


IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']


def create_image(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        form = ImageForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            image = form.save(commit=False)
            image.user = request.user
            image.image_file = request.FILES['image_file']
            file_type = image.image_file.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in IMAGE_FILE_TYPES:
                context = {
                    'image': image,
                    'form': form,
                    'error_message': 'Image file must be PNG, JPG, or JPEG'
                }
                return render(request, 'convert/create_image.html', context)
            image.save()
            images = Image.objects.filter(user=request.user)
            return render(request, 'convert/images.html', {'images': images})
        context = {
            'form': form,
        }
        return render(request, 'convert/create_image.html', context)


def create_filter(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        form = FilterForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            filter = form.save(commit=False)
            filter.user = request.user
            filter.filter_file = request.FILES['filter_file']
            file_type = filter.filter_file.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in IMAGE_FILE_TYPES:
                context = {
                    'filter': filter,
                    'form': form,
                    'error_message': 'Image file must be PNG, JPG, or JPEG'
                }
                return render(request, 'convert/create_filter.html', context)
            filter.save()
            filters = Filter.objects.filter(user=request.user)
            return render(request, 'convert/filters.html', {'filters': filters})
        context = {
            'form': form,
        }
        return render(request, 'convert/create_filter.html', context)


# def create_session(request):
#     if not request.user.is_authenticated():
#         return render(request, 'convert/login.html')
#     else:
#         form = SessionForm(request.POST or None)
#         if form.is_valid():
#             image = form.save(commit=False)
#             image.user = request.user
#             image.save()
#             return render(request, 'convert/detail_session.html', {'image': image})
#         context = {
#             'form': form,
#         }
#         return render(request, 'convert/create_session.html', context)


def delete_filter(request, filter_id):
    filter = Filter.objects.get(pk=filter_id)
    filter.delete()
    filters = Filter.objects.filter(user=request.user)
    return render(request, 'convert/filters.html', {'filters': filters})


def delete_image(request, image_id):
    image = Image.objects.get(pk=image_id)
    image.delete()
    images = Image.objects.filter(user=request.user)
    return render(request, 'convert/images.html', {'images': images})


# def delete_session(request, filter_id, image_id, session_id):
#     filter = get_object_or_404(Filter, pk=filter_id)
#     image = get_object_or_404(Image, pk=image_id)
#     session = Session.objects.get(pk=session_id)
#     session.delete()
#    return render(request, 'convert/sessions.html', {'filter': filter, 'image': image})


class UserFormView(View):
    form_class = UserForm
    template_name = 'convert/visitor.html'

    # Use this function when it is get request
    def get(self, request):
        form = self.form_class(None)
        return render(request, self.template_name, {'form': form})

    # Use this function when it is post request
    def post(self, request):
        form = self.form_class(request.POST)

        if form.is_valid():
            user = form.save(commit=False)

            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user.set_password(password)
            user.save()

            user = authenticate(username=username, password=password)

            if user is not None:
                if user.is_active:
                    login(request, user)
                    return redirect('gallery:user')
                    # get user's info: request.user.username / etc.

        return render(request, self.template_name, {'form': form})


def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return render(request, 'convert/user.html')
            else:
                return render(request, 'convert/login.html', {'error_message': 'Your account has been disabled'})
        else:
            return render(request, 'convert/login.html', {'error_message': 'Invalid login'})
    return render(request, 'convert/login.html')


def logout_user(request):
    logout(request)

    return render(request, 'convert/visitor.html')

