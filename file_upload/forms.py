from django import forms
from django.core.files.storage import default_storage

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
    
    # def save(self, requests):
    #     # if requests.method=="POST":
    #     upload_file=self.cleaned_data['file']
    #     file_name=default_storage.save(upload_file.name, upload_file)
    #     return default_storage.url(file_name)
