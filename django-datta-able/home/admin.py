from django.contrib import admin

from django.apps import apps
from .models import UploadedImage

# Register your models here.
admin.site.register(UploadedImage)

app_models = apps.get_app_config('home').get_models()
for model in app_models:
    try:    

        admin.site.register(model)

    except Exception:
        pass
