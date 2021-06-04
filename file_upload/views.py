from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from django.http import HttpResponse
# from .models import Image, Image_DL


import sys

# models.pyとImageFieldを使ったやり方模索中
#参照：https://hisafi.hatenablog.com/entry/2017/07/09/212430
# def file_upload(request):
#     if request.method=="POST":
#         form = Image(request.POST)

#Ajax trial
def test_ajax_app(request):
    hoge="Hello Django!!"
    return render(request, "file_upload/upload.html", {"hoge":hoge,})

def test_ajax_response(request):
    input_text=request.POST.getlist("name_input_text")
    # ↑ name_input_textというname属性を持つinputタグに入力されたデータを取り出している。
    hoge="Ajax Response" + input_text[0]
    return HttpResponse(hoge)
    # ↑ 部分的なHTMLとして返すには、HttpResponse()を使う。
        
# ------------------------------------------------------------------
def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            sys.stderr.write("*** file_upload *** aaa ***\n")
            handle_uploaded_file(request.FILES['file'])
            file_obj = request.FILES['file']
            sys.stderr.write(file_obj.name + "\n")
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'file_upload/upload.html', {'form': form})
#
#
# ------------------------------------------------------------------
def handle_uploaded_file(file_obj):
    sys.stderr.write("*** handle_uploaded_file *** aaa ***\n")
    sys.stderr.write(file_obj.name + "\n")
    file_path = 'media/documents/' + file_obj.name 
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