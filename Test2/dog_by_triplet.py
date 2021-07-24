# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:23:07 2021

@author: 81805
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import load_dataset2 as ld
from sklearn.manifold import TSNE
import logging
import datetime
import logging.handlers

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from kerastuner.tuners import RandomSearch
import IPython

from pca_plotter import PCAPlotter

#loggerを設定
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#Handlerを設定
ima=datetime.datetime.now()
filename="./Log/Log_file {0} {1}h-{2}m-{3}s.log".format(ima.strftime('%Y-%m-%d'), ima.hour, ima.minute, ima.second)
h1=logging.handlers.RotatingFileHandler(filename, maxBytes=10000, backupCount=10)
h1.setLevel(logging.DEBUG)
#HandlerのFormatを設定
formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
#loggerにHandlerを設定
logger.addHandler(h1)
print('TensorFlow version:', tf.__version__)

# print("Please load train dataset")
# (x_train, y_train) = ld.load_dataset(normalize=False, flatten=True)

# print('Please load test dataset')
# (x_test, y_test) =ld.load_dataset(normalize=False, flatten=True)

# Loading the Data

(x_train, y_train), (x_test, y_test), size = ld.main()
logger.debug('x_train : {}'.format(x_train))
logger.debug('x_train.shape : {}'.format(x_train.shape))
logger.debug('y_train : {}'.format(y_train))
logger.debug('y_train.shape : {}'.format(y_train.shape))
logger.debug('x_test : {}'.format(x_test))
logger.debug('x_test.shape : {}'.format(x_test.shape))
logger.debug('y_test : {}'.format(y_test))
logger.debug('y_test.shape : {}'.format(y_test.shape))
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# size = 28
print("Image size is ", size)
Image_size=size*size
channel = 3

# print('x_train.shape 1 : ', x_train.shape)
# print('y_train.shape 1 :', y_train.shape)
# print('y_train : ', y_train)

x_train = np.reshape(x_train, (x_train.shape[0], size, size, channel))/255.0
x_test = np.reshape(x_test, (x_test.shape[0], size, size, channel))/255.0 # 画像サイズの件は、後で調整する
logger.debug('x_train.shape : {}'.format(x_train.shape))
logger.debug('y_train.shape : {}'.format(y_train.shape))
logger.debug('x_train.shape[0] : {}'.format(x_train.shape[0]))
logger.debug('y_train : {}'.format(y_train))
print('y_train : {}'.format(y_train))

# Plotting Examples

def plot_triplets(examples):
    logger.debug('Start plot_triplets')
    plt.figure(figsize=(24, 8))
    for i in range(3):
        logger.debug("example[i].shape : {}".format(examples[i].shape))
        plt.subplot(1,3, 1+ i)
        plt.imshow(np.reshape(examples[i], (size, size, channel)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    logger.debug('End plot_triplets')

plot_triplets([x_train[0], x_train[1], x_train[2]])
# plot_triplets(x_train[0])

# def plot_tsne(x, y, colormap=plt.cm.Paired):
#     print(x.shape)
#     print(y.shape)
#     x=np.reshape(x, [x.shape[0], x.shape[1]])

#     print(x.shape)
#     # x=np.reshape(x, (x.shape[0], x.shape[1]))
#     # y=np.reshape(y, (y.shape[0], y.shape[1]))
#     plt.figure(figsize=(8, 6))
    
#     #clean the figure
#     plt.clf()
    
#     tsne = TSNE(n_components=2)
#     X_embedded = tsne.fit_transform(x)
#     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)
    
#     plt.xticks(())
#     plt.yticks(())
    
#     plt.show()

# plot_tsne(x_train, y_train)

# A Batch of Triplets

def create_batch(batch_size=10):
    logger.debug('Start create_batch')
    logger.debug('create_batch : Part A')
    x_anchors=np.zeros((batch_size, size, size, channel))
    x_positives=np.zeros((batch_size, size, size, channel))
    x_negatives=np.zeros((batch_size, size, size, channel))
    # x_anchors=np.zeros((batch_size, Image_size), dtype=np.float32)
    # x_positives=np.zeros((batch_size, Image_size), dtype=np.float32)
    # x_negatives=np.zeros((batch_size, Image_size), dtype=np.float32)
    # print("x_anchors : ", x_anchors, x_anchors.shape)
    # print("x_positives : ", x_positives, x_positives.shape)
    # print("x_negatives : ", x_negatives, x_negatives.shape)
    logger.debug('Part A completed')
    
    logger.debug('create_batch : Part B')
    # try:
    a=0
    for i in range(0, batch_size):
        a+=1
        random_index = random.randint(0, x_train.shape[0]-1)
        logger.debug('random_index : {}'.format(random_index))
        x_anchor = x_train[random_index]
        y = y_train[random_index]
        logger.debug('x_anchor : {}'.format(x_anchor))
        logger.debug('x_anchor.shape : {}'.format(x_anchor.shape))
        logger.debug('y_anchor : {}'.format(y))
        
        # plt.figure(figsize=(24, 8))
        # for c in range(3):
        #     print(x_anchor.shape)
        #     plt.subplot(1,3, 1+c)
        #     plt.imshow(np.reshape(x_anchor, (size, size)), cmap='binary')
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()
        
        # TypeErrorなどが出たときに、try/exceptで中身を出せないか？
        # np.squeezeで、配列が消えて、整数だけになってしまうケースがあるみたい。→TypeErrorの原因。これはなぜか？
        # batch_size=1で、create_batchを回すと比較的スムーズ。batch_sizeの設定方法を見直す必要あり？
        
        dogs_for_pos = np.squeeze(np.where(y_train==y))
        dogs_for_neg = np.squeeze(np.where(y_train!=y))
        logger.debug("before squeeze (pos) : {}".format(np.where(y_train==y)))
        logger.debug('before squeeze (neg) : {}'.format(np.where(y_train!=y)))
        logger.debug('dogs_for_pos : {}'.format(dogs_for_pos))
        logger.debug('dogs_for_pos.shape : {}'.format(dogs_for_pos.shape))
        logger.debug('dogs_for_neg : {}'.format(dogs_for_neg))
        logger.debug('dogs_for_neg.shape : {}'.format(dogs_for_neg.shape))
        
        # print("dogs_for_pos : ", dogs_for_pos)
        
        x_positive = x_train[dogs_for_pos[random.randint(0, len(dogs_for_pos)-1)]]
        x_negative = x_train[dogs_for_neg[random.randint(0, len(dogs_for_neg)-1)]]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        logger.debug('Part B completed {}'.format(a))
    return [x_anchors, x_positives, x_negatives]
    
    # except:
    #     print('y : ', y)
    #     print('y shape :', y.shape)
    #     print('x_positives[i] shape : ', x_positives[i].shape)
    #     print('x_negatives[i] shape : ', x_negatives[i].shape)
    #     print('positive_shape', x_positive.shape)
    #     print('negative_shape', x_negative.shape)
    #     print('x_positive', x_positive)
    #     print('x_negative', x_negative)
    #     print("random_index : ", random_index)
    #     print("dog_for_pos : ", dogs_for_pos)
    #     print("dog_for_neg : ", dogs_for_neg)
    #     print('dog_for_pos shape : ', dogs_for_pos.shape)
    #     print('dog_for_neg shape : ', dogs_for_neg.shape)
    #     print(np.where(y_train==y))
    #     print(np.where(y_train!=y))


examples=create_batch(1)
plot_triplets(examples)


# Embedding Model
# logger.debug('Embedding Model')
emb_size=32

# embedding_model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(Image_size,)),
#     tf.keras.layers.Dense(emb_size, activation='sigmoid')
#     ])
# embedding_model.summary()
# logger.debug('Complete')
# embedding_model=tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(4, kernel_size=(3,3), activation='relu', input_shape=(size, size, channel)),
#     tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
#     tf.keras.layers.Conv2D(16, kernel_size = (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
#     tf.keras.layers.Conv2D(32, kernel_size= (5,5), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
#     tf.keras.layers.Dropout(0.25),
#     # tf.keras.layers.Flatten(),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(64, activation = 'relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(emb_size, activation = 'sigmoid')])

# embedding_model=embedding_model()
# embedding_model.summary()

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

def build_model(hp):
    model = keras.Sequential()
    hp_units = hp.Int('units', min_value = 16, max_value = 512, step=32)
    # filters=hp.Choice('num_filters', value=[16, 32, 64, 128, 256], default=64)
    model.add(Conv2D(filters=hp.Int('convolution_1', min_value=16, max_value=512, step=32), kernel_size=hp.Choice('convolution_1', values=[3,6]), activation='relu', input_shape=(size, size, channel), padding='same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=hp.Int('convolution_2', min_value=16, max_value=512, step=32), kernel_size=hp.Choice('convolution_2', values=[3,6]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=hp.Int('convolution_3', min_value=16, max_value=512, step=32), kernel_size=hp.Choice('convolution_3', values=[3,6]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=hp_units, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='test_dir', project_name='helloworld')
tuner.search_space_summary()
tuner.search(x=x_train, y=y_train, epochs=3, validation_data=(x_test, y_test))
tuner.results_summary()

best_hps=tuner.get_best_hyperparameters(num_trials = 1)[0]
print(best_hps)
embedding_model=tuner.hypermodel.build(best_hps)
# embedding_model=tuner.get_best_models(num_models=1)

example=np.expand_dims(x_train[0], axis=0)
# example=x_train[0]
# print("x_train[0] : ", x_train[0])
# print("x_train[0].shape : ", x_train[0].shape)
# print("example : ", example)
# print("example.shape : ", example.shape)
example_emb=embedding_model.predict(example)[0]
# print("example_emb", example_emb)
# print("example_emb.shape : ", example_emb.shape)

# CNN test (2021/07/18)
# model = Sequential()
# def CNN_model():
#     model.add(Conv2D(64, kernel_size = (5,5), activation='relu', input_shape = (size, size, 1)))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(Conv2D(128, kernel_size = (5,5), activation ='relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(Conv2D(128, kernel_size = (5,5), activation ='relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(Flatten())
#     model.add(Dense(256, activation = 'relu'))
#     model.add(Dense(emb_size, activation = 'sigmoid'))
    

# embedding_model=CNN_model()
# embedding_model.summary()


# Siamese Network
logger.debug('Siamese Network')
input_anchor = tf.keras.layers.Input(shape=(size, size, channel))
input_positive = tf.keras.layers.Input(shape=(size, size, channel))
input_negative = tf.keras.layers.Input(shape=(size, size, channel))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

# output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative])

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()
logger.debug('completed')

# Triplet Loss

alpha = 0.2

def triplet_loss(y_true, y_pred):
    logger.debug('Triplet Loss')
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2*emb_size], y_pred[:, 2*emb_size]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    logger.debug('completed')
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


# Data Generator
def data_generator(batch_size=10):
    logger.debug('Data Generator')
    b=0
    while True:
        b+=1
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        logger.debug('Data Generating {}'.format(b))
        yield x, y
    logger.debug('completed')


# class LossHistory(tf.keras.callbacks.Callback):
#     def __init__(self):
#         # コンストラクタに保持用の配列を宣言しておく
#         self.train_acc = []
#         self.train_loss = []
#         self.val_acc = []
#         self.val_loss =[]
    
#     def on_epoch_end(self, epoch, logs={}):
#         # 配列にEpochが終わるたびにappendしていく
#         self.train_acc.append(logs['acc'])
#         self.val_acc.append(logs['val_acc'])
#         self.train_loss.append(logs['loss'])
#         self.val_loss.append(logs['val_loss'])
        
#         # グラフ描画部分
#         plt.figure(num=1, clear=True)
#         plt.title('accuracy')
#         plt.xlabel('epoch')
#         plt.ylabel('accuracy')
#         plt.plot(self.train_acc, label='train')
#         plt.plot(self.val_acc, label='validation')
#         plt.legend()
#         plt.pause(0.1)
        
# cb_my = LossHistory()



# Model Training
logger.debug('Model Training')
batch_size=1
epochs = 10
steps_per_epoch = int(x_train.shape[0]/batch_size)
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
es_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, mode='auto')
red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2, verbose=1)


net.compile(loss=triplet_loss, metrics=['acc'], optimizer=opt)

_=net.fit(
    data_generator(batch_size), 
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=1,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            x_test[:100], y_test[:100]), es_cb, red_lr]          
)

# print(_.history)
loss = _.history['loss']
# # val_loss = _.history['val_loss']

# learning_count = len(loss) + 1

# plt.plot(range(1, learning_count), loss, marker = '+', label = 'loss')
# # plt.plot(range(1, learning_count), val_loss, marker = ".", label = "val_loss")
# plt.legend(loc="best", fontsize = 10)
# plt.xlabel("learning_count")
# plt.ylabel("loss")
# plt.show()

accuracy = _.history["acc"]
# val_accuracy = _.hisotry["val_accuracy"]

learning_count=len(accuracy) + 1

plt.plot(range(1, learning_count), accuracy, marker="+", label="accuracy")
plt.plot(range(1, learning_count), loss, marker = '.', label = 'loss')
# plt.plot(range(1, learning_count), val_accuracy, marker=".", label="val_accuracy")
plt.legend(loc="best", fontsize=10)
plt.xlabel("laearning_count")
plt.ylabel("accuracy")
plt.show()

print('Every process completed')









