# -*- coding: utf-8 -*-
"""
create on 2018-5-22 20:59:44
by：weifeng

"""

import imp
import random
import numpy as np
import pandas as pd
import scipy as sp

import Evaluation.Evaluation as Evaluation
import LFM.LFM as LFM
from ItemBase import ItemCF, ItemCF_IUF
from UserBase import UserCF,UserCF_IIF
from Implicit import ALS_WR,BPR

imp.reload(UserCF)
imp.reload(ItemCF)
imp.reload(ItemCF_IUF)
imp.reload(Evaluation)
imp.reload(LFM)


def readData():
    ###采用fast_FM数据
    dataHome = './testdata/'
    small_data = pd.read_csv(filepath_or_buffer=dataHome + 'small_data.csv',encoding="UTF-8")
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
    ##采用的是u.data数据
    # data = []
    # user_totalcount = {}      ###user对应的所有的播放次数，用播放次数来衡量用户的兴趣度，如果用户的播放次数越多，说明用户对这个物品越喜欢
    # with open(dataHome + 'u.data','r') as fr:
    #     for line in fr.readlines():
    #         lineArr = line.strip().split()
    #         user = lineArr[0]
    #         artist = lineArr[1]
    #         count = int(lineArr[2])
    #
    #         user_totalcount.setdefault(user,0)
    #         user_totalcount[user] += count
    # with open(dataHome + 'u.data','r') as frr:
    #     for line in frr.readlines():
    #         curline = line.strip().split()
    #         user = curline[0]
    #         artist = curline[1]
    #         count = int(curline[2])
    #         data.append([user,artist,count/user_totalcount[user] *1.0])
    # return data

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

#################################
#####为了测试BPR，ALS_WR模型
#################################
def readImplicitData():
    dataHome = './testdata/'
    small_data = pd.read_csv(filepath_or_buffer=dataHome + 'small_data.csv',encoding="UTF-8")
    small_data['user'] = small_data['user'].astype("category")
    small_data['artist'] = small_data['artist'].astype("category")
                                                                                        ###共计100000条数据
    plays = sp.sparse.coo_matrix((small_data['plays'].astype(float),                         ###943 * 1682
            (small_data['user'].cat.codes, small_data['artist'].cat.codes)), dtype = np.double)   ####这个矩阵构造的就是用户的矩阵
    ##如果使用的原来的数据
    # small_data = pd.read_csv(filepath_or_buffer=dataHome + 'u.data',encoding="UTF-8",
    #                          sep='\t', header=None,
    #                          usecols =[0,1,2],
    #                          names=['user', 'artist', 'plays'])
    # small_data['user'] = small_data['user'].astype("category")
    # small_data['artist'] = small_data['artist'].astype("category")
    #                                                                                     ###共计100000条数据
    # plays = sp.sparse.coo_matrix((small_data['plays'].astype(float),                         ###943 * 1682
    #         (small_data['user'].cat.codes, small_data['artist'].cat.codes)), dtype = np.double)   ####这个矩阵构造的就是用户的矩阵

    artist_cat = small_data['artist'].cat.codes
    return plays,artist_cat
def split_train_test(plays, train_rate=0.8):   ###这里的训练集表示的是每一个用户的比例
    user_index = range(plays.shape[0])  ###用户的总数
    train = plays.copy().tolil()
    test = sp.sparse.lil_matrix(plays.shape)  ##构造了和play同样大小的matrix ,只不过为空

    min_rows = int(1 / (1 - train_rate))  ##rate = 5
    for uindex in user_index:   ###[0....943]  ###对于具体的用户
        rows = plays.getrow(uindex).indices
        if len(rows) <= min_rows:    ##不能低于最小的行数
            continue
        testindics = np.random.choice(plays.getrow(uindex).indices,
                                      size=int(len(rows) * (1 - train_rate)),
                                      replace=False)
        train[uindex, testindics] = 0.  ##在训练集上，抽样出来的索引，用户对物品的关系，设为空
        test[uindex, testindics] = plays[uindex, testindics]  ###在测试集上，抽样出来的索引用户对物品的关系，保持原样，相当于每个用户都采样了相同的比例
                                                               ###但是在测试集合中，和训练集合相比，没被选中的索引，为原来的值
    train = train.tocsr()
    train.eliminate_zeros()  ###去除零，目前只剩下有作用的物品
    return train, test.tocsr()
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
    
        W_user = UserCF_IIF.UserSimilarity(train)       ###基于用户的协同过滤
        # rank = UserCF_IIF.Recommend('1',train,W)
        result_user = UserCF_IIF.Recommendation(test.keys(), train, W_user)
        
    #    W = ItemCF.ItemSimilarity(train)
    #    rank = ItemCF.Recommend('1',train,W)
#        result =  ItemCF.Recommendation(test.keys(),train, W)
        
        W_item = ItemCF_IUF.ItemSimilarity(train)      ###基于物品的协同过滤
    #    rank = ItemCF_IUF.Recommend('1',train,W)
        result_item =  ItemCF_IUF.Recommendation(test.keys(),train, W_item)

        [P,Q] = LFM.LatentFactorModel(train, 15,40, 0.02, 0.01) ###使用隐语义模型
        # rank = LFM.Recommend('2',train,P,Q)
        result_LFM = LFM.Recommendation(test.keys(), train,P,Q)

        precision = {}
        recall = {}
        coverage ={}
        popularity={}
        N = 10
        for str,result in {"user_CF":result_user,"item_CF":result_item, "LFM":result_LFM}.items():
                precision.setdefault(str,0)
                recall.setdefault(str,0)
                coverage.setdefault(str,0)
                popularity.setdefault(str,0)
                precision[str] +=Evaluation.Precision(train,test, result,N)
                recall[str] += Evaluation.Recall(train,test,result,N)
                coverage[str] += Evaluation.Coverage(train, test, result,N)
                popularity[str] += Evaluation.Popularity(train, test, result,N)
    numFlod =1
    for str in ["user_CF", "item_CF", "LFM"]:
        precision[str] /= numFlod
        recall[str] /= numFlod
        coverage[str] /= numFlod
        popularity[str] /= numFlod

    # 输出结果
    for str in ["user_CF","item_CF","LFM"]:
        print('%s precision = %f'%(str,precision[str]))
        print('%s recall = %f'%(str,recall[str]))
        print('%s coverage = %f'%(str,coverage[str]))
        print('%s popularity = %f'%(str,popularity[str]))
    ####测试隐式反馈 BPR
    plays,artist_cat = readImplicitData()                                           ##生成隐式反馈的训练数据和测试数据
    train_mat,test_mat = split_train_test(plays.tocsr())

    bprmodel = BPR.train_pair_wise_model_and_evaluate(train_mat, test=test_mat)     ##测试BPR模型
    BPR.BPR_Evaluation(bprmodel, train_mat, test_mat)

    # bpr_recommender = BPR.BPR_Recommender(bprmodel, plays.tocsr(), artist_cat)    ##使用BPR进行推荐
    # recommendation1=bpr_recommender.recommend(25,top=20, with_history=False)
    # print(recommendation1)

    ####测试ALS_WR,网格搜索
    score =0
    factor_most =80
    regularization_most =0.01
    epochs_most = 80
    # for factor in [20,40,60,80,100]:                                             ##使用网格搜索选择最合适的参数
    #     for regularization in [0.1,1,5,10,20]:
    #         for epochs in [10,20,40,60,80]:
    #             alsmodel,auc = ALS_WR.train_point_wise_model_and_evaluate(train_mat, plays, test_mat, factors=factor,
    #                                                                       regularization=regularization, epochs=epochs,
    #                                                                       num_test_users=100)
    #             if score <auc:
    #                 score = auc
    #                 factor_most=factor
    #                 regularization_most=regularization
    #                 epochs_most=epochs
    print("best score",score)
    print("best parameter",factor_most,regularization_most,epochs_most)
    alsmodel, auc = ALS_WR.train_point_wise_model_and_evaluate(train_mat, plays, test_mat, factors=factor_most, ##使用ALS_WR
                                                               regularization=regularization_most, epochs=epochs_most,
                                                               num_test_users=100)
    ALS_WR.ALS_WR_Evaluation(alsmodel, train_mat.tocsr(), train_mat, test_mat)

    # als_recommender = ALS_WR.ALS_Recommender(alsmodel, plays.tocsr(), artist_cat)                     ##使用ALS_WR推荐
    # recommendation2 = als_recommender.recommend(25,top=20, with_history=False)
    # print(recommendation2)