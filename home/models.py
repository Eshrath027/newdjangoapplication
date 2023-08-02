from django.db import models

# Create your models here.

class UploadedImage(models.Model):
    Name=models.CharField(max_length=20, null=True)
    Age=models.CharField(max_length=20, null=True)
    Gender=models.CharField(max_length=20, null=True)
    image = models.ImageField(upload_to='uploaded_images/')
    tumor_image = models.ImageField(upload_to='tumor_images/', null=True)
    masked_image = models.ImageField(upload_to='mask_images/', null=True)
    predicted_class = models.CharField(max_length=20, null=True)
    Area = models.CharField(max_length=20, null=True)
    Volume = models.CharField(max_length=20, null=True)
    Stage = models.CharField(max_length=20, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
class Product(models.Model):
    id    = models.AutoField(primary_key=True)
    name  = models.CharField(max_length = 100) 
    info  = models.CharField(max_length = 100, default = '')
    price = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.name
