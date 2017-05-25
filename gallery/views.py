from django.contrib.auth import authenticate, login, logout
from django.db.models import Q
from django.http import JsonResponse
from django.views.generic import View
from django.shortcuts import render, redirect
from .models import Image, Filter, Session
from .forms import UserForm, ImageForm, FilterForm, SessionForm

# Neural Style Dependencies
import numpy as np
import scipy.misc
from .stylize import stylize
import math
from argparse import ArgumentParser
from PIL import Image as PILImage

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'gallery\imagenet-vgg-verydeep-19.mat'
POOLING = 'max'


# Neural Style Functions
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('content', help='content image', metavar='CONTENT')
    parser.add_argument('styles', nargs='+', help='one or more style images', metavar='STYLE')
    parser.add_argument('output', help='output path', metavar='OUTPUT')
    parser.add_argument('iterations', type=int, help='iterations (default %(default)s)', metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    return parser


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    PILImage.fromarray(img).save(path, quality=95)


# Django View Functions
def userview(request):
    if not request.user.is_authenticated():
        return render(request, 'convert/login.html')
    else:
        return render(request, 'convert/user.html')


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

            # Create separate function for download-progress view
            # filter_id = request.POST.get('filter')
            # image_id = request.POST.get('image')
            # output_title = request.POST.get('output_image_title')
            # output_image_path = 'gallery/static/convert/output/' + output_title
            # iterations = request.POST.get('iterations')

            # filter = Filter.objects.get(pk=int(filter_id))
            # image = Image.objects.get(pk=int(image_id))

            # parser = ns.build_parser()
            # options = parser.parse_args(['gallery/' + image.image_file.url[1:], 'gallery/' + filter.filter_file.url[1:], output_image_path, str(iterations)])
            # ns.main(options)

            session.save()

            session_id = session.id
            # session_input = Session.objects.get(pk=session_id)

            return redirect(str(session_id) + '/progress/')
            # return render(request, 'convert/conversion_progress.html', {'session': session_input})

        context = {
            'form': form,
        }

        return render(request, 'convert/sessions.html', context)


ITERATION = 0


def progress(request, session_id):
    session = Session.objects.get(pk=session_id)

    # filter_id = session.filter
    # image_id = session.image

    filter = session.filter
    image = session.image
    output_title = session.output_image_title
    output_image_path = 'gallery/static/convert/output/' + output_title
    iterations = session.iterations

    # filter = Filter.objects.get(pk=int(filter_id))
    # image = Image.objects.get(pk=int(image_id))

    perm = request.GET.get('q')

    if perm:
        parser = build_parser()
        options = parser.parse_args(['gallery/' + image.image_file.url[1:], 'gallery/' + filter.filter_file.url[1:],
                                     output_image_path, str(iterations)])

        content_image = imread(options.content)
        style_images = [imread(style) for style in options.styles]

        width = options.width
        if width is not None:
            new_shape = (int(math.floor(float(content_image.shape[0]) /
                                        content_image.shape[1] * width)), width)
            content_image = scipy.misc.imresize(content_image, new_shape)
        target_shape = content_image.shape
        for i in range(len(style_images)):
            style_scale = STYLE_SCALE
            if options.style_scales is not None:
                style_scale = options.style_scales[i]
            style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                                                  target_shape[1] / style_images[i].shape[1])

        style_blend_weights = options.style_blend_weights
        if style_blend_weights is None:
            # default is equal weights
            style_blend_weights = [1.0 / len(style_images) for _ in style_images]
        else:
            total_blend_weight = sum(style_blend_weights)
            style_blend_weights = [weight / total_blend_weight
                                   for weight in style_blend_weights]

        initial = options.initial
        if initial is not None:
            initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
            # Initial guess is specified, but not noiseblend - no noise should be blended
            if options.initial_noiseblend is None:
                options.initial_noiseblend = 0.0
        else:
            # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
            if options.initial_noiseblend is None:
                options.initial_noiseblend = 1.0
            if options.initial_noiseblend < 1.0:
                initial = content_image

        # if options.checkpoint_output and "%s" not in options.checkpoint_output:
        #     parser.error("To save intermediate images, the checkpoint output "
        #                  "parameter must contain `%s` (e.g. `foo%s.jpg`)")

        for ITERATION, image in stylize(
            network=options.network,
            initial=initial,
            initial_noiseblend=options.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=options.preserve_colors,
            iterations=options.iterations,
            content_weight=options.content_weight,
            content_weight_blend=options.content_weight_blend,
            style_weight=options.style_weight,
            style_layer_weight_exp=options.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            beta1=options.beta1,
            beta2=options.beta2,
            epsilon=options.epsilon,
            pooling=options.pooling,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations
        ):
            output_file = None
            combined_rgb = image
            if ITERATION is not None:
                if options.checkpoint_output:
                    output_file = options.checkpoint_output % ITERATION
                    # yield iteration
            else:
                output_file = options.output
            if output_file:
                imsave(output_file, combined_rgb)

        return render(request, 'convert/download.html', {'session': session})
    else:
        return render(request, 'convert/conversion_progress.html', {'session': session})

    # return render(request, 'convert/download.html', {'session': session})


def update_progress(request, session_id):
    session = Session.objects.get(pk=session_id)

    return JsonResponse({'curr_iter': CONTENT_WEIGHT_BLEND, 'maxm_iter': session.iterations})


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
    form = UserForm(request.POST or None)
    context = {'form': form}

    return render(request, 'convert/visitor.html', context)

