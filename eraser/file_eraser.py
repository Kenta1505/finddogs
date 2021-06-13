import glob
import os
import schedule
import time
from config.settings import BASE_DIR as b

dir_name=b + "/static"

def file_eraser():
    # if os.path.isfile('testsite2/media/*.xlsx'):
    xlsx_list=glob.glob(dir_name)
    print(xlsx_list)
    for file_xlsx in xlsx_list:
        os.remove(file_xlsx)
    # if os.path.isfile('testsite2/*.txt'):
    txt_list=glob.glob(dir_name)
    print(txt_list)
    for file_txt in txt_list:
        os.remove(file_txt)
    # if os.path.isfile('testsite2/*.log'):
    log_list=glob.glob(dir_name)
    print(log_list)
    for file_log in log_list:
        os.remove(file_log)

def main():
    schedule.every(1).minutes.do(
        file_eraser
    )
    while True:
        #loggingを使えば、ここにも処理前後などのメッセージを入れられるのでは？print()でもいいのかも？？
        schedule.run_pending()
        time.sleep(1)

if __name__=="__main__":
    main()
