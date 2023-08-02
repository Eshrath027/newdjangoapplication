from django.shortcuts import render, redirect
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout
import os
import matplotlib.pyplot as plt
import io
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile

from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.models import load_model
from .models import UploadedImage
from ultralytics import YOLO
from IPython import display
display.clear_output()
import boto3
import tensorflow as tf
import ultralytics
import h5py
import torch 

import s3fs
import zipfile
import tempfile
import numpy as np
from tensorflow import keras
from pathlib import Path
import logging

def index(request):
  context = {
    'segment': 'index'
  }
  return render(request, "pages/index.html", context)

def tables(request):
  context = {
    'segment': 'tables'
  }
  return render(request, "pages/tables.html", context)

# Components
@login_required(login_url='/accounts/login/')
def bc_button(request):
  context = {
    'parent': 'basic_components',
    'segment': 'button'
  }
  return render(request, "pages/components/bc_button.html", context)

@login_required(login_url='/accounts/login/')
def bc_badges(request):
  context = {
    'parent': 'basic_components',
    'segment': 'badges'
  }
  return render(request, "pages/components/bc_badges.html", context)

@login_required(login_url='/accounts/login/')
def bc_breadcrumb_pagination(request):
  context = {
    'parent': 'basic_components',
    'segment': 'breadcrumbs_&_pagination'
  }
  return render(request, "pages/components/bc_breadcrumb-pagination.html", context)

@login_required(login_url='/accounts/login/')
def bc_collapse(request):
  context = {
    'parent': 'basic_components',
    'segment': 'collapse'
  }
  return render(request, "pages/components/bc_collapse.html", context)

@login_required(login_url='/accounts/login/')
def bc_tabs(request):
  context = {
    'parent': 'basic_components',
    'segment': 'navs_&_tabs'
  }
  return render(request, "pages/components/bc_tabs.html", context)

@login_required(login_url='/accounts/login/')
def bc_typography(request):
  context = {
    'parent': 'basic_components',
    'segment': 'typography'
  }
  return render(request, "pages/components/bc_typography.html", context)

@login_required(login_url='/accounts/login/')
def icon_feather(request):
  context = {
    'parent': 'basic_components',
    'segment': 'feather_icon'
  }
  return render(request, "pages/components/icon-feather.html", context)


# Forms and Tables
@login_required(login_url='/accounts/login/')
def form_elements(request):
  context = {
    'parent': 'form_components',
    'segment': 'form_elements'
  }
  return render(request, 'pages/form_elements.html', context)

@login_required(login_url='/accounts/login/')
def basic_tables(request):
  context = {
    'parent': 'tables',
    'segment': 'basic_tables'
  }
  return render(request, 'pages/tbl_bootstrap.html', context)

# Chart and Maps
@login_required(login_url='/accounts/login/')
def morris_chart(request):
  context = {
    'parent': 'chart',
    'segment': 'morris_chart'
  }
  return render(request, 'pages/chart-morris.html', context)

@login_required(login_url='/accounts/login/')
def google_maps(request):
  context = {
    'parent': 'maps',
    'segment': 'google_maps'
  }
  return render(request, 'pages/map-google.html', context)

# Authentication
class UserRegistrationView(CreateView):
  template_name = 'accounts/auth-signup.html'
  form_class = RegistrationForm
  success_url = '/accounts/login/'

class UserLoginView(LoginView):
  template_name = 'accounts/auth-signin.html'
  form_class = LoginForm

class UserPasswordResetView(PasswordResetView):
  template_name = 'accounts/auth-reset-password.html'
  form_class = UserPasswordResetForm

class UserPasswrodResetConfirmView(PasswordResetConfirmView):
  template_name = 'accounts/auth-password-reset-confirm.html'
  form_class = UserSetPasswordForm

class UserPasswordChangeView(PasswordChangeView):
  template_name = 'accounts/auth-change-password.html'
  form_class = UserPasswordChangeForm

def logout_view(request):
  logout(request)
  return redirect('/accounts/login/')

@login_required(login_url='/accounts/login/')
def profile(request):
  context = {
    'segment': 'profile',
  }
  return render(request, 'pages/profile.html', context)

@login_required(login_url='/accounts/login/')
def sample_page(request):
  context = {
    'segment': 'sample_page',
  }
  return render(request, 'pages/sample-page.html', context)


@login_required(login_url='/accounts/login/')
def detect_tumor(request):
    c=UploadedImage.objects.count()
    last_uploaded_image = None 
    if request.method == 'POST' and request.FILES['image']:
        name = request.POST.get('name')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)

        
        AWS_ACCESS_KEY = "AKIARDLKIIDKFLFLXA5N"
        AWS_SECRET_KEY = "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"
        BUCKET_NAME = "newtumormodel"
        MODEL_KEY3= "pf3new_newbrats_3d.hdf5"
        MODEL_KEY1= "3-conv-128-nodes-1-dense-model.h5"
        # MODEL_KEY2= "best (1).pt"


# Initialize S3 client
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

        # Load the model from S3 directly without downloading to local path
        model_file3 = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY3)
        model_content3 = model_file3['Body'].read()

        # Use io.BytesIO to create a buffer to load the model
        # Use h5py to load the model
        with h5py.File(io.BytesIO(model_content3), 'r') as f:
            model3 = load_model(f,compile=False)
        #model1
        model_file1 = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY1)
        model_content1 = model_file1['Body'].read()

        # Use io.BytesIO to create a buffer to load the model
        # Use h5py to load the model
        with h5py.File(io.BytesIO(model_content1), 'r') as f:
            model1 = load_model(f,compile=False)
        # Load your pre-trained models
        #detection_model = load_model('path/to/your/detection_model.h5')
        # model = load_model(r"home\3-conv-128-nodes-1-dense-model.h5",compile=False)
        model2 = YOLO(r'home\best (1).pt')
        # model3 = load_model(r"home\pf3new_newbrats_3d.hdf5",compile=False)


        #MODEL_3
        img = Image.open(image_file).convert('L')  # Read the image in grayscale

        # Resize the 2D image to 128x128
        resized_image= img.resize((128, 128))
        depth = 128
        img_3d = np.stack([resized_image] * depth, axis=-1)
        temp_combined_images = np.stack((img_3d,)*3, axis=-1)
        test_img=temp_combined_images
        test_img_input = np.expand_dims(test_img, axis=0)
        test_prediction = model3.predict(test_img_input)
        test_prediction_argmax = np.argmax(test_prediction, axis=4)[0]
        print(test_prediction_argmax.shape)
        voxel_volume =0.1
        # Calculate the volume of the tumor
        tumor_volume = np.sum(test_prediction_argmax == 1) * voxel_volume
        print(tumor_volume)
        n_slice = 55
        mask_image_array=test_prediction_argmax[:,:, n_slice]
        image_data_uint8 = mask_image_array.astype(np.uint8)
        #mask_image= Image.fromarray(image_data_uint8)
        image_save_path = os.path.join('mask_image', 'maskimage'+str(c+1)+'.jpg')
        #mask_image.save(image_save_path)
        plt.imsave(image_save_path, image_data_uint8)
        #mask_image = self.cleaned_data['mask_image'] 
   

        #AREA CODE
        pixel_size = 0.1  # Replace with the actual pixel size or scale of your image
        unique_values = np.unique(mask_image_array)
        segment_areas = {}
        for segment_value in unique_values:
            # Create a binary mask for the current segment
            binary_mask = (mask_image_array == segment_value).astype(np.uint8)
            pixels_count = np.sum(binary_mask)
            area = pixels_count * pixel_size * pixel_size
            segment_areas[segment_value] = area
        total_area = sum(segment_areas.values())
        #for segment_value, area in segment_areas.items():
        #    print("Area of segmented part with value", segment_value, ":", area, "units squared")
        stage=""
        for segment_value, area in segment_areas.items():
          seg=total_area-area
          break 
        s = round(seg)
        if s in range(0,8):
          stage="Stage-1"
        elif s in range(8,14):
          stage="Stage-2"
        elif s in range(14,21):
          stage="Stage-3"
        else:
          stage="Stage-4"

        #MODEL_1
        image = Image.open(image_file).convert('RGB')
        #image_raw = Image.open(image_file)
        image = image.resize((224, 224))  # Assuming your model expects 224x224 images
        x = np.array(image)
        x = x.reshape(1, 224, 224, 3)  # Assuming RGB images
        x=x/255.0
        prediction = model1.predict(x)
        y_bool = np.argmax(prediction, axis=1)
        z=y_bool[0]
        if z==0:
          pc="Glioma"
        elif z==1:
          pc="Meningioma"
        elif z==2:
          pc="no_tumor"
        elif z==3:
          pc="pituitary"      
        
        model2(image,save =True)
        # Read the segmented image and store it in the database as tumor_image
        no_tumor_volume=0
        no_seg=0
        no_stage= "Nil"
        if pc!="no_tumor":
          uploaded_image = UploadedImage(Name=name, Age=age, Gender=gender, image=image_file, predicted_class=pc,Volume=str(round(tumor_volume,2))+' mm cubic units', Area=str(round(seg,2) )+' mm square units', Stage=stage)
          uploaded_image.save()
          
          segmented_image_path = os.path.join('runs', 'segment', 'predict'+str(c+2), 'image0.jpg')
          with open(segmented_image_path, 'rb') as f:
              uploaded_image.tumor_image.save('segmented_image.jpg', f)

          image_retrieve_path = os.path.join('mask_image', 'maskimage'+str(c+1)+'.jpg')    
          with open(image_retrieve_path, 'rb') as f:
              uploaded_image.masked_image.save('mask_image.jpg', f)
        else:
          uploaded_image = UploadedImage(Name=name, Age=str(age), Gender=gender,image=image_file, predicted_class=pc, tumor_image=image_file,masked_image=image_file, Volume=no_tumor_volume, Area=no_seg, Stage=no_stage)
          uploaded_image.save() 

        last_uploaded_image = UploadedImage.objects.latest('uploaded_at')
        



    return render(request, 'pages/detect_tumor.html', {'last_uploaded_image': last_uploaded_image})
    #return render(request, 'pages/detect_tumor.html', {'uploaded_images': uploaded_images})



