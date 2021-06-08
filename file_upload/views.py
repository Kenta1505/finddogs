from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from django.http import HttpResponse
# from .models import Image, Image_DL
# from .models import ImageUpload
from django.core.files.storage import default_storage
import os
from PIL import Image
import sys

# models.pyとImageFieldを使ったやり方模索中
#参照：https://hisafi.hatenablog.com/entry/2017/07/09/212430
# def file_upload(request):
#     if request.method=="POST":
#         form = Image(request.POST)

#Ajax trial

file_path=""


def test_ajax_app(request):
    if request.method == 'POST':
        # form=ImageUpload(request.POST, request.FILES)
        title=str(request.POST['title'])
        hoge="Hello Django!!" + title
        # img=request.FILES['file']
        # print(img)
        # context={
        #     "hoge":hoge,
        #     "form":img,
        # }
        # return render(request, "file_upload/upload.html", context)
    
        return HttpResponse(hoge)
        
    else:
        form=UploadFileForm()
        return render(request, 'file_upload/upload.html', {'form': form})


def test_ajax_response(request):
    if not request.method=="POST":
        form = UploadFileForm()
        return render(request, 'file_upload/upload.html', {'form': form})
        # input_text=request.POST.getlist("name_input_text")
        # ↑ name_input_textというname属性を持つinputタグに入力されたデータを取り出している。
        # print(input_text)
    else:
        form = UploadFileForm(request.POST, request.FILES)
        sys.stderr.write("*** file_upload *** aaa ***\n")
        handle_uploaded_file(request.FILES['file'])
        file_obj = request.FILES['file']
        sys.stderr.write(file_obj.name + "\n")
        # return str(file_obj.name)

        file_name=file_obj.name
        file="images/" + file_name
        # file="images/" + str(request.FILES['file'])
        # files=os.listdir("images")
        context=render(request, "file_upload/upload.html", {"form":form})
        hoge="Ajax Response" + str(request.POST['title'])
        # file_path=os.listdir("/../images")
        print(hoge, file)
        img=Image.open(file)
        img.show()
        return HttpResponse(hoge, file, context)
            

# ------------------------------------------------------------------
def file_upload(request):
    if request.method=="POST":

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            sys.stderr.write("*** file_upload *** aaa ***\n")
            handle_uploaded_file(request.FILES['file'])
            file_obj = request.FILES['file']
            sys.stderr.write(file_obj.name + "\n")
            # return str(file_obj.name)
            print("file_upload HttpResponseRedirectです。")
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    print("file_upload render部分です。")
    return render(request, 'file_upload/upload.html', {'form': form})
# #
# #
# # ------------------------------------------------------------------
def handle_uploaded_file(file_obj):
    sys.stderr.write("*** handle_uploaded_file *** aaa ***\n")
    sys.stderr.write(file_obj.name + "\n")
    file_path = 'images/' + file_obj.name 
    sys.stderr.write(file_path + "\n")
    with open(file_path, 'wb+') as destination:
        for chunk in file_obj.chunks():
            sys.stderr.write("*** handle_uploaded_file *** ccc ***\n")
            destination.write(chunk)
            sys.stderr.write("*** handle_uploaded_file *** eee ***\n")
#
# # ------------------------------------------------------------------
# def success(request):
#     str_out = "Success!<p />"
#     str_out += "成功<p />"
    
#     return HttpResponse(str_out)
# ------------------------------------------------------------------