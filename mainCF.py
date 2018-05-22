# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:36:48 2018

@author: lanlandetian
"""

import UserCF
import UserCF_IIF
import ItemCF
import ItemCF_IUF
import random
import Evaluation
import LFM

import numpy as np
import pandas as pd

import imp

imp.reload(UserCF)
imp.reload(ItemCF)
imp.reload(ItemCF_IUF)
imp.reload(Evaluation)
imp.reload(LFM)


def readData():
    ###采用fast_FM数据
    small_data = pd.read_csv(filepath_or_buffer='./small_data.csv',encoding="UTF-8")
    small_data['user'] = small_data['user'].astype("category")
    small_data['artist'] = small_data['artist'].astype("category")
    user_artist = list(zip(list(small_data['user'].cat.codes),list(small_data['artist'].cat.codes)))
    user_artist_count = list(zip(user_artist,list(small_data['plays'])))
    del small_data
    data =[]
    user_totalcount ={}
    for item in user_artist_count:
        user = item[0][0]
        count = item[1]
        user_totalcount.setdefault(user,0)
        user_totalcount[user] += int(count)
    for item in user_artist_count:
        user = item[0][0]
        artist = item[0][1]
        count = int(item[1])
        data.append([user,artist,count/user_totalcount[user]])
    return data
    ###采用的是u.data数据
    # data = []
    # fileName = './u.data'
    # user_totalcount = {}      ###user对应的所有的播放次数，用播放次数来衡量用户的兴趣度，如果用户的播放次数越多，说明用户对这个物品越喜欢
    # with open('./u.data','r') as fr:
    #     for line in fr.readlines():
    #         lineArr = line.strip().split()
    #         user = lineArr[0]
    #         artist = lineArr[1]
    #         count = int(lineArr[2])
    #
    #         user_totalcount.setdefault(user,0)
    #         user_totalcount[user] += count
    # with open('./u.data','r') as frr:
    #     for line in frr.readlines():
    #         curline = line.strip().split()
    #         user = curline[0]
    #         artist = curline[1]
    #         count = int(curline[2])
    #         data.append([user,artist,count/user_totalcount[user] *1.0])

    # for line in fr.readlines():
    #     lineArr = line.strip().split()              ##这里没有用到用户的听歌次数,只要死用户听过这个歌手的歌曲，我就认为这个用户对这个歌手的作用为1
    #     data.append([lineArr[0], lineArr[1], 1.0])  ##每一次都添加了一个列表，data是一个二维数组
    # return data                                      ##这里只要用户消费了物品，就认为用户对物品喜爱
    
###抽出来的用户基本上差不多，差的也就是用户的
def SplitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for user, item,rating in data:
        if random.randint(0,M-1) == k:   ###在这个地方由于 测试集合 1/K
            test.append([user,item,rating])   ###在这个地方，其实分割数据也是测试集和训练集中用户和物品都有
        else:
            train.append([user, item,rating])
    return train, test
        
    
# 将列表形式数据转换为dict形式
def transform(oriData):
    ret = dict()
    for user,item,rating in oriData:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating  ##这是一个二维的形式，转换为user 对Item的评分
    return ret
    
if __name__ == '__main__':
    data = readData()
    numFlod = 5    ###定义几折验证
    precision =0
    recall = 0
    coverage = 0
    popularity =0
    for i in range(0,1):
        [oriTrain,oriTest] = SplitData(data,numFlod,i,0)
        train = transform(oriTrain)
        test = transform(oriTest)
        
#        W = UserCF.UserSimilarity(train)
    #    rank = UserCF.Recommend('1',train,W)
#        result = UserCF.Recommendation(test.keys(), train, W)
    
        # W = UserCF_IIF.UserSimilarity(train)
    #    rank = UserCF_IIF.Recommend('1',train,W)
    #     result = UserCF_IIF.Recommendation(test.keys(), train, W)
        
    #    W = ItemCF.ItemSimilarity(train)
    #    rank = ItemCF.Recommend('1',train,W)
#        result =  ItemCF_IUF.Recommendation(test.keys(),train, W)
        
        # W = ItemCF_IUF.ItemSimilarity(train)
    #    rank = ItemCF_IUF.Recommend('1',train,W)
    #     result =  ItemCF_IUF.Recommendation(test.keys(),train, W)

        [P,Q] = LFM.LatentFactorModel(train, 15,40, 0.02, 0.01) ###
        #rank = LFM.Recommend('2',train,P,Q)
        result = LFM.Recommendation(test.keys(), train,P,Q)


        N = 10
        precision += Evaluation.Precision(train,test, result,N)
        recall += Evaluation.Recall(train,test,result,N)
        coverage += Evaluation.Coverage(train, test, result,N)
        popularity += Evaluation.Popularity(train, test, result,N)
       
    # precision /= numFlod
    # recall /= numFlod
    # coverage /= numFlod
    # popularity /= numFlod

    precision /= 1
    recall /= 1
    coverage /= 1
    popularity /= 1
    
     #输出结果
    print('precision = %f' %precision)
    print('recall = %f' %recall)
    print('coverage = %f' %coverage)
    print('popularity = %f' %popularity)
    