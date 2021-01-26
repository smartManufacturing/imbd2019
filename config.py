
class _config:
        #train 資料夾位置
        Train_Path='./train/'
        #test 資料夾位置
        Test_Path='./test/'
        #img size
        imgrow=164
        imgcol=4
        #PTC 類別定義
        PTC_class = {'G11': 0, 'G15': 1, 'G17':2, 'G19':3,'G32':4,'G34':5,'G48':6,'G49':7}
        test_y={'1.txt':'G11','2.txt':'G11',
                '3.txt':'G15','4.txt':'G15','5.txt':'G15','6.txt':'G15','7.txt':'G15','8.txt':'G15',
                '9.txt':'G17','10.txt':'G17',
                '11.txt':'G19','12.txt':'G19',
                '13.txt':'G32','14.txt':'G32','15.txt':'G32','16.txt':'G32','17.txt':'G32','18.txt':'G32',
                '19.txt':'G34','20.txt':'G34','21.txt':'G34','22.txt':'G34','23.txt':'G34','24.txt':'G34',
                '25.txt':'G48','26.txt':'G48','27.txt':'G48','28.txt':'G48','29.txt':'G48','30.txt':'G48',
                '31.txt':'G49','32.txt':'G49','33.txt':'G49','34.txt':'G49','35.txt':'G49','36.txt':'G49',
                }
        Batch_size=32
        epoch=70
        class_num=8

