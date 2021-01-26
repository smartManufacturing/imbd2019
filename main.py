from file import _file
from config import _config
from Mymodel import CNN_Model,LSTM_Model
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tkinter import font
import os

def Project_Classification():
        tXt=_file()
        Set=_config()
        #讀取train data 及 test data
        train_x,train_y,trainfilename=tXt.ReadData_TxT(Set.Train_Path)
        test_x,test_y,testfilename=tXt.ReadData_TxT(Set.Test_Path)
        print('=======',train_x.shape,train_y.shape)
        # print(testfilename)
        Real_y=list()
        for i in testfilename:
                _class=Set.test_y[i]
                Real_y.append(Set.PTC_class[_class])
        CNN=CNN_Model()
        Predict_Y=CNN.Classification(train_x,train_y,test_x,test_y)
        # goolgenet=goolgenet_Model()
        # goolgenet.Classification(train_x,train_y,test_x,test_y)
        # LSTM=LSTM_Model()
        # Predict_Y=LSTM.Classification(train_x,train_y,test_x,test_y)


        Real_yNp=np.array(Real_y)
        print('predict:', Predict_Y )
        print('Real:   ',Real_yNp )
        Result=0
        for i in range(Real_yNp.shape[0]):
                if(Predict_Y[i]==Real_yNp[i]):
                        Result=Result+1
        Error=Result/Real_yNp.shape[0]
        print('Accuracy(%):',Error*100)


def main():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0

        Project_Classification()


if __name__ == "__main__":
    main()

