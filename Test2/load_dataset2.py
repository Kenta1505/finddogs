# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:52:15 2021

@author: 81805

https://qiita.com/seamcarving/items/87384ac0dab2bc2c4c15

"""

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os, glob

base_dir="./Dataset2"

def my_load_data(folder_str, size):
    print("load_dataset...")
    print(folder_str)
    folders=folder_str.split(',')
    print(folders)
    c=0
    for dirs in folders:
        dirs = base_dir + "/" + dirs
        c += sum(os.path.isfile(os.path.join(dirs, name)) for name in os.listdir(dirs))
        print('{}'.format(dirs), os.listdir(dirs))
    X = []
    Y = []
    a=0
    for index, fol_name in enumerate(folders):
        a+=1
        # print(a)
        # print(index)
        # print(fol_name)
        files = glob.glob(base_dir + "/" + fol_name + "/*.jpg")
        for file in files:
            image = Image.open(file)
            if size=='':
                size=int(image.size[0])
            else:
                size=int(size)
            image = image.resize((size, size))
            # print(image.size)
            # image = image.convert('L')
            data = np.asarray(image)
            X.append(data)
            Y.append(fol_name)
    X = np.array(X)
    Y = np.array(Y)
    # print("X", X)
    # print("Y", Y)
    # oh_encoder=OneHotEncoder(categories='auto', sparse=False)
    # onehot=oh_encoder.fit_transform(pd.DataFrame(Y))
    # x_train, x_test, y_train, y_test = train_test_split(X, onehot, test_size=0.2)
    lb_encoder=LabelEncoder()
    lb=lb_encoder.fit_transform(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, lb, test_size=0.2)
    print("Total Number of files -------------", c)
    print("Number of x_test : ", len(x_test))
    print("Number of x_train : ", len(x_train))
    return x_train, x_test, y_train, y_test, size

import argparse

def main():
    # parser = argparse.ArgumentParser(description='sample')
    # parser.add_argument('--folder', '-i')
    # parser.add_argument('--size', '-s', type=int, default=28)
    # args = parser.parse_args()
    # x_train, x_test, y_train, y_test = my_load_data(args.folder, args.size)    
    
    folder = base_dir
    folder=os.listdir(folder)
    folder=','.join(folder)
    print(folder)
    
    # folder = input("Please input folder_names with ',' (ex. f1,f2,f3)___")
    size = input('Please input file_size or press enter___')
    # if size == '':
    #     size=28
    x_train, x_test, y_train, y_test, size = my_load_data(folder, size)
    print(x_train.dtype)
    print(y_train.dtype)
    # print(np.finfo(x_train))
    # print(np.finfo(y_train))
    x_train=x_train.astype(np.float32)
    y_train=y_train.astype(np.float32)
    print(x_train.dtype, y_train.dtype)
    
    # print("x_train", x_train)
    # print('y_train', y_train)
    return(x_train, y_train), (x_test, y_test), size

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test), size = main()
    # print('x_train :', x_train, 'x_test : ', x_test)
    # print('y_train : ', y_train, 'y_test : ', y_test)
    
    
    
    
    
    