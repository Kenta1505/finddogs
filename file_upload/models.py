from django.db import models

class ImageUpload(models.Model):
    image=models.ImageField(upload_to='../images', blank=True, null=True)
    title=models.CharField(max_length=32)