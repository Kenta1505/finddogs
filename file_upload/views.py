from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from django.http import HttpResponse
# from .models import Image, Image_DL
# from .models import ImageUpload
from django.core.files.storage import default_storage

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
        file_path=file_upload(request, UploadFileForm(request.POST, request.FILES))
        print(file_path)
        # img=request.FILES['file']
        # print(img)
        # context={
        #     "hoge":hoge,
        #     "form":img,
        # }
        # return render(request, "file_upload/upload.html", context)
    
        return HttpResponse(hoge, file_path)
        
    else:
        form=UploadFileForm()
        return render(request, 'file_upload/upload.html', {'form': form})


def test_ajax_response(request):
    input_text=request.POST.getlist("name_input_text")
    # ↑ name_input_textというname属性を持つinputタグに入力されたデータを取り出している。
    print(input_text)
    hoge="Ajax Response" + str(request.POST['title'])
    file_path="images/" + str(file_upload(request, UploadFileForm(request.POST, request.FILES)))
    print(hoge, file_path)
    return HttpResponse(hoge, file_path)
    # ↑ 部分的なHTMLとして返すには、HttpResponse()を使う。
        
# ------------------------------------------------------------------
def file_upload(request,form):
    # if request.method == 'POST':
    # form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
        sys.stderr.write("*** file_upload *** aaa ***\n")
        handle_uploaded_file(request.FILES['file'])
        file_obj = request.FILES['file']
        sys.stderr.write(file_obj.name + "\n")
        return file_obj
            # return HttpResponseRedirect('/success/url/')
    # else:
    #     form = UploadFileForm()
    # return render(request, 'file_upload/upload.html', {'form': form})
#
#
# ------------------------------------------------------------------
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
# ------------------------------------------------------------------
def success(request):
    str_out = "Success!<p />"
    str_out += "成功<p />"
    
    return HttpResponse(str_out)
# ------------------------------------------------------------------