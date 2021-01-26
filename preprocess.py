import numpy as np
class _preprocess:
        def __init__(self):
                pass

        def splitdata(self,dataNp,ratio,imgrow):
                #input資料row維度
                imgcounter=int(dataNp.shape[0]/imgrow)
                #train維度=all*ratio
                trainCounter=int(imgcounter*ratio)
                trainNp=dataNp[0:trainCounter*imgrow,:]


                #test=all-train
                testNp=dataNp[trainCounter*imgrow:imgcounter*imgrow,:]

                return trainNp,testNp



