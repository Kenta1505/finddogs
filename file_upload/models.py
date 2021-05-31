from django.db import models
# import requests
# import os

# # models.pyとImageFieldを使ったやり方模索中
# #参照：https://hisafi.hatenablog.com/entry/2017/07/09/212430
# class Image(models.Model):
#     image=models.ImageField(upload_to="image/")

# class Image_DL():
#     def save_and_rename(self, url, name=None):
#         res=requests.get(url)
#         if res.status_code != 200:
#             return 'No Image'
#         path =os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/MyApp/media/image/"
#         if name==None:
#             path += url.split("/")[-1]
#         else:
#             path += name
#         with open(path, "wb") as file:
#             file.write(res.content)
#         return path