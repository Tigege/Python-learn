import numpy as np

def KNN(X_train,y_train,X_test,k):
    #修改列表为numpy类型
    X_train=np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    #获得训练、测试数据的长度
    X_train_len=len(X_train)
    X_test_len=len(X_test)
    pre_lable=[]  #存储预测标签

    '''
    依次遍历测试数据，计算每个测试数据与训练数据的距离值，排序，
    根据前K个投票选举出预测结果
    '''
    for test_len in range(X_test_len):  #计算测试的第一组数据
        dis=[]
        for train_len in range(X_train_len):
            temp_dis=abs(sum(X_train[train_len,:]-X_test[test_len,:]))#计算距离
            dis.append(temp_dis**0.5)
        '''
         print(dis)
               train
               [[ 1  2  3  4]
               [ 5  6  7  8]
               [ 9 10 11 12]
               [ 1  2  3  4]
               [ 5  6  7  8]
               [ 9 10 11 12]]
               test
               [[1 2 3 4]
                [5 6 7 8]]
               dis
               [0.0, 4.0, 5.656854249492381,0.0, 4.0, 5.656854249492381]
               [4.0, 0.0, 4.0,4.0, 0.0, 4.0]
               '''
        dis=np.array(dis)
        sort_id=dis.argsort()
        # 按照升序进行快速排序，返回的是原数组的下标。
        # 比如，x = [30, 10, 20, 40]
        # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
        # 那么，numpy.argsort(x) = [1, 2, 0, 3]
        '''
        print(sort_id)
        [0 3 1 4 2 5]
        [1 4 0 2 3 5]
        '''
        '''
        dicc1=[]
        dicc2=[]
        flag=1
        for i in range(k):
            for len1 in range(len(dicc1)):
                print(len1)
                if(len(dicc1)==0):
                    break
                if(y_train[sort_id[i]]==dicc1[len1]):
                    dicc2[len1]=dicc2[len1]+1
                    flag=0
            if(flag==1):
                dicc1.append(y_train[sort_id[i]])
                dicc2.append(1)
                flag=1
        print("test>>>")
        print(dicc1)
        print(dicc2)
        max=0
        temp=0
        for i in range(len(dicc2)):
            if(max<dicc2[i]):
                max=dicc2[i]
                temp=dicc1[i]
        pre_lable.append(temp)
        #那么训练集标签中的对应位置的标签既为预测标签
        '''

        dic={}
        for i in range(k):
            vlable=y_train[sort_id[i]]  #为对应的标签记数
            dic[vlable]=dic.get(vlable,0)+1
            #寻找vlable代表的标签，如果没有返回0并加一，如果已经存在返回改键值对应的值并加一
        max = 0
        for index, v in dic.items():   #.items  返回所有的键值对
            if v > max:
                max = v
                maxIndex = index
        pre_lable.append(maxIndex)

    print(X_train)
    print("test")
    print(X_test)
    print("---------")

    print(y_train)
    print("pre")
    print(pre_lable)
    '''
        C:\python_64\python.exe F:/Githouse/ML_learning/KNN.py
        [[ 1  2  3  4]
         [ 5  6  7  8]
         [ 9 10 11 12]
         [ 1  2  3  4]
         [ 5  6  7  8]
         [ 9 10 11 12]]
        test
        [[1 2 3 4]
         [5 6 7 8]]
        ---------
        [1 2 3 1 2 3]
        pre
        [1, 2]
    '''

if __name__=="__main__":
    '''
    X_train=[[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12]]

    y_train=[1,2,3]

    X_test=[[1,2,3,4],   #那么预测数据应为 1、2
            [5,6,7,8]]
    '''

    X_train = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

    y_train = [1, 2, 3,1,2,3]

    X_test = [[1, 2, 3, 4],  # 那么预测数据应为 1、2
              [5, 6, 7, 8]]
    KNN(X_train,y_train,X_test,2)