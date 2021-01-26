import sys
from os import walk
import numpy as np
import cv2
from config import _config  
class _file:


        def __init__(self):
                self.Set=_config()
                pass


        def ReadData_TxT(self,Path):
                AllPTCNp=np.zeros((1,self.Set.imgcol))
                AllDataList=list()
                labellsit=list()
                PTClabelList=list()
                namelist=list()
                #讀取資料夾內檔案
                for root,dirs,file_all in walk(Path):
                        
                        for _file in dirs:
                                #讀取各類別資料夾(G11、G15、G17、G19、G32、G34、G48、G49)
                                for root,dirs,file_all in walk(Path+_file):
                                        #讀取各類別資料夾內的TXT
                                        for name in file_all:
                                                # print(name)
                                                namelist.append(name)
                                                Datalist=list()
                                                SingleTxTNp=np.zeros((1,4))
                                                PTCList=list()
                                                StartSaveData=False

                                                #讀取TXT
                                                f=open(Path+_file+'/'+name,'r',encoding='big5')
                                                data=f.readlines()
                                                for i in data:
                                                        Datalist.append(i)
                                                f.close()
                                                
                                                
                                                
                                                #讀取TXT內容後，淨化資料內容
                                                for i in Datalist:
                                                        #判斷此TXT類別
                                                        if i.find('G')>-1:
                                                                label=i[i.find('G'):i.find('-')]
                                                                
                                                                labellsit.append(label)
                                                        #經過Deg.F 確認開始讀取 PTC
                                                        if StartSaveData:
                                                                PTCList.append(i)

                                                                #若讀到換行符號，row方向往下堆疊
                                                                if i.find('\n')>=0:
                                                                        PTCList=PTCList[0].split('\t')
                                                                        showNp=np.array(PTCList)
                                                                        
                                                                        PTCList=[]
                                                                        if showNp.shape[0]>1: #txt開頭有換行符號則須避開                                                                               
                                                                                SingleTxTNp=np.vstack((SingleTxTNp,showNp[0:self.Set.imgcol]))#資料往下堆疊
                                                        if i.find('Deg.F')>-1:#經過Deg.F 確認開始讀取 PTC
                                                                StartSaveData=True
                                                
                                                
                                                SingleTxTNp=SingleTxTNp[1:self.Set.imgrow,:]#去除SingleTxTNp row=0資料，只留實際PTC資料
                                                SingleTxTNp_float=SingleTxTNp.astype(np.float)#convert string array to float

                                                AllPTCNp=np.vstack((AllPTCNp,SingleTxTNp_float))#讀完整個txt內容後存入結果 
                                ResultNp=AllPTCNp[1:AllPTCNp.shape[0],:].copy()#去除AllPTCNp row=0資料，只留實際PTC資料
                                


                #儲存Label資料
                for i in labellsit:
                        i=i.strip(' ')
                        PTClabelList.append(self.Set.PTC_class[i])
                PTClabelNp=np.array([PTClabelList])
                PTClabelNp=PTClabelNp.T
                print(ResultNp.shape)

                return ResultNp,PTClabelNp,namelist

