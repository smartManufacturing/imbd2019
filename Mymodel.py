from __future__ import print_function
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input,Conv2D, MaxPooling2D,AveragePooling2D, Flatten , Conv1D, MaxPooling1D, GlobalAveragePooling1D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from keras.optimizers import RMSprop, Adam, Adadelta
from preprocess import _preprocess
from config import _config
import math
import matplotlib.pyplot as plt


from keras.utils.np_utils import to_categorical
import pickle, gzip, urllib.request, json



def plothistory(history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

# 繪製訓練 & 驗證的損失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()  

class CNN_Model:

        def __init__(self):

                pass


        def Classification(self,x_train_all_Np,y_train_all_Np,x_testNp,y_testNp):
                Set=_config()
                process=_preprocess()

                #train data分80%,test data分20%
                train_x_np,val_x_np=process.splitdata(x_train_all_Np,0.8,1)
                train_y_np,val_y_np=process.splitdata(y_train_all_Np,0.8,1)
                print(train_x_np.shape,val_x_np.shape,train_y_np.shape,val_y_np.shape)

                # input image dimensions
                img_rows, img_cols = Set.imgrow-1,Set.imgcol
                train_x_np=train_x_np.reshape(int(train_x_np.shape[0]/(img_rows)),img_rows,img_cols,1)
                val_x_np=val_x_np.reshape(int(val_x_np.shape[0]/(img_rows)),img_rows,img_cols,1)
                x_train_all_Np=x_train_all_Np.reshape(int(x_train_all_Np.shape[0]/(img_rows)),img_rows,img_cols,1)
                #print(x_train_all_Np.shape,x_testNp.shape)
                
                _input_shape=(img_rows,img_cols,1)

                #Normalize
                x_mean=np.mean(x_train_all_Np,axis=0)
                x_std=np.std(x_train_all_Np,axis=0)
                
                train_x_np=(train_x_np-x_mean)/x_std
                val_x_np=(val_x_np-x_mean)/x_std
               
                print('x_train shape:',x_train_all_Np.shape)
                print(train_x_np.shape ,'train samples')
                print(val_x_np.shape ,'val samples')
                print(train_y_np.shape ,'trainy samples')
                print(val_y_np.shape ,'valy samples')
                print(train_y_np)
                train_y_np_Onehot=keras.utils.to_categorical(train_y_np,Set.class_num)
                val_y_np_Onehot=keras.utils.to_categorical(val_y_np, Set.class_num)
                print('train_y_np_Onehot',train_y_np_Onehot.shape,'val_y_np_Onehot',val_y_np_Onehot.shape)

                print('_input_shape',_input_shape)
                model = Sequential()
                model.add(Conv2D(64, kernel_size=(1, 1),
                                         activation='relu',
                                        #  padding='SAME',
                                        input_shape=_input_shape))
                # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#
                # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#

                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))
                # # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#
                # # model.add(Conv2D(64, kernel_size=(2, 2),activation='relu'))#

                # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#


                model.add(MaxPooling2D(pool_size=(2, 2)))
                # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#

                # model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))#

                

                # model.add(MaxPooling2D(pool_size=(2, 2)))#



                # model.add(Conv2D(64, (3, 3), activation='relu'))
                # model.add(MaxPooling2D(pool_size=(2, 2)))
                #model.add(Dropout(0.25))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(256, activation='relu'))
                # model.add(Dense(512, activation='relu'))
                model.add(Dense(512, activation='relu'))
                # model.add(Dense(1024, activation='relu'))
                model.add(Dense(1024, activation='relu'))
                model.add(Dense(2048, activation='relu'))
                # model.add(Dense(2048, activation='relu'))
                # model.add(Dense(2048, activation='relu'))
                # model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(Set.class_num, activation='softmax'))

                model.summary()
                model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer='adam',
                                metrics=['accuracy'])

                # model.fit(x_train_all_Np, y_train_all_Np_Onehot, validation_split=0.2, batch_size=32, epochs=50)
                print('t',val_x_np.shape,val_y_np_Onehot.shape)

                history=model.fit(train_x_np,
                        train_y_np_Onehot,
                        # validation_split=0.5,
                        batch_size=Set.Batch_size,
                        epochs=Set.epoch,
                        #verbose=1)
                        validation_data=(val_x_np,val_y_np_Onehot ))
                         # score = model.evaluate(x_testNp, y_testNp_Onehot)
                score = model.evaluate(val_x_np, val_y_np_Onehot, verbose=0)

                x_testNp=x_testNp.reshape(int(x_testNp.shape[0]/(img_rows)),img_rows,img_cols,1)
                x_testNp=(x_testNp-x_mean)/x_std

                predction=model.predict_classes(x_testNp)
                # print(predction)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                # plothistory(history)



                
                return predction

import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation,Bidirectional
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

class LSTM_Model:
        def __init__(self):
                pass
        def Classification(self,x_train_all_Np,y_train_all_Np,x_testNp,y_testNp):
                Set=_config()
                process=_preprocess()
                # learning_rate = 0.001
                training_iters = Set.epoch
                batch_size = Set.Batch_size
                display_step = 10

                n_input = Set.imgcol
                n_step = Set.imgrow-1 
                n_hidden = 256
                n_classes = Set.class_num

                train_x_np,val_x_np=process.splitdata(x_train_all_Np,0.8,1)
                train_y_np,val_y_np=process.splitdata(y_train_all_Np,0.8,1)

 
                
                x_train_all_Np=x_train_all_Np.reshape(-1, n_step, n_input)
                train_x_np = train_x_np.reshape(-1, n_step, n_input)
                val_x_np = val_x_np.reshape(-1, n_step, n_input)
               

                # train_x_np = train_x_np.astype('float32')
                # val_x_np = val_x_np.astype('float32')



                x_mean=np.mean(x_train_all_Np,axis=0)
                x_std=np.std(x_train_all_Np,axis=0)
                
                train_x_np=(train_x_np-x_mean)/x_std
                val_x_np=(val_x_np-x_mean)/x_std



     
                train_y_np_Onehot = keras.utils.to_categorical(train_y_np, n_classes)
                val_y_np_Onehot = keras.utils.to_categorical(val_y_np, n_classes)

                model = Sequential()
                model.add(LSTM(n_hidden,batch_input_shape=(None, n_step, n_input),unroll=True))
                # model.add(LSTM(n_hidden,unroll=True))
                # model.add(LSTM(n_hidden))

                # model.add(LSTM(n_hidden,batch_input_shape=(None, n_step, n_input),unroll=True))

                # model.add(Bidirectional(LSTM(units=2,return_sequences=False)))
                model.add(Dropout(0.5))
                model.add(Dense(n_classes))
                model.add(Activation('softmax'))

                adam = Adam(lr=0.001)
                model.summary()
                model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

                model.fit(train_x_np, train_y_np_Onehot,
                batch_size=batch_size,
                epochs=training_iters,
                verbose=1,
                validation_data=(val_x_np, val_y_np_Onehot))

                scores = model.evaluate(val_x_np, val_y_np_Onehot, verbose=2)
                print('LSTM test score:', scores[0])
                print('LSTM test accuracy:', scores[1])


                x_testNp=x_testNp.reshape(-1, n_step, n_input)
                x_testNp=(x_testNp-x_mean)/x_std
                predction=model.predict_classes(x_testNp)
                print(predction)
                return predction


# class goolgenet_Model:

#         def __init__(self):
#                 pass

        

#         def Classification(self,x_train_all_Np,y_train_all_Np,x_testNp,y_testNp):
#                 #coding=utf-8
#                 Set=_config()
#                 process=_preprocess()
#                 seed = 7
#                 np.random.seed(seed)
# # Load the dataset
#                 # Set=_config()
#                 # process=_preprocess()
#                 # urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
#                 # with gzip.open('mnist.pkl.gz', 'rb') as f:
#                 #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#                 # print('====',train_set[0].shape,valid_set[0].shape,test_set[0].shape)
# #train_x is [0,1]

#                 #train data分80%,test data分20%
#                 train_x_np,val_x_np=process.splitdata(x_train_all_Np,0.8,1)
#                 train_y_np,val_y_np=process.splitdata(y_train_all_Np,0.8,1)

#                 img_rows, img_cols = Set.imgrow-1,Set.imgcol
#                 train_x = train_x_np.reshape((-1,img_rows,img_cols,1))
#                 train_y = keras.utils.to_categorical(train_y_np,Set.class_num)
 
#                 valid_x = val_x_np.reshape((-1,img_rows,img_cols,1))
#                 valid_y = keras.utils.to_categorical(train_y_np,Set.class_num)
 
#                 test_x = x_testNp.reshape((-1,img_rows,img_cols,1))
                
 
#                 x_train_all_Np=x_train_all_Np.reshape((-1,img_rows,img_cols,1))


#                 #Normalize
#                 x_mean=np.mean(x_train_all_Np,axis=0)
#                 x_std=np.std(x_train_all_Np,axis=0)
#                 train_x=(train_x-x_mean)/x_std
#                 valid_x=(valid_x-x_mean)/x_std
#                 test_x=(test_x-x_mean)/x_std


#                 model = Sequential()
#                 model.add(Conv2D(32,(1,1),strides=(1,1),input_shape=(img_rows,img_cols,1),padding='valid',activation='relu',kernel_initializer='uniform'))
#                 model.add(MaxPooling2D(pool_size=(2,2)))
#                 model.add(Conv2D(64,(1,1),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
#                 model.add(MaxPooling2D(pool_size=(2,2)))
#                 model.add(Flatten())
#                 model.add(Dense(100,activation='relu'))
#                 model.add(Dense(10,activation='softmax'))
#                 model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
#                 # model.summary()
 
#                 # model.fit(train_x,train_y,validation_data=(valid_x,valid_y),batch_size=20,epochs=20,verbose=2)
#                 # score = model.evaluate(valid_x, valid_y, batch_size=20,verbose=2)
#                 # predction=model.predict_classes(x_testNp)
#                 # print(predction)

       

