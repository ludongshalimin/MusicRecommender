# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:04:07 2018

@author: lanlandetian
"""

import math

def GetRecommendation(result, user, N = 5000):  ##从用户排名的列表中，得到用户排名前十的物品
    rank = result[user]  ##获取user的结果
    ret = []
    if len(rank)  > N:   ##如果给用户待推荐的结果大于N，这地方是否是应该改成[:N],因为只对应了前N个推荐
        for item,rating in rank[:10]:
            ret.append((item,rating))
    else:               ###小于N直接返回了
        ret = rank
    return ret
    
def Recall(train,test,result,N = 5000):
    hit = 0
    all = 0
    for user in test.keys():
        tu = test[user] ###u在测试集上作用的物品
        rank = GetRecommendation(result, user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)
    
def Precision(train, test,result, N = 5000):
    hit = 0
    all = 0
    for user in test.keys():
        tu = test[user]  ##tu,代表了用户在测试集上作用的物品
        rank = GetRecommendation(result,user,N)
        for item, pui in rank:
            if item in tu:   ##如果推荐的列表中，有在测试集上出现，说明用户消费了这个物品，代表了是一个成功的推荐
                hit += 1
        all += len(rank)
    return hit / (all * 1.0)  ###最终确定被用户消费的，占推荐的比例
    
def Coverage(train, test, result, N = 5000):
    recommend_items = set() ##利用测试集的user,进行推荐所得到的所有的推荐物品的列表
    all_items = set()  ##训练集中，所有的物品
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
            
    for user in test.keys():
        rank = GetRecommendation(result,user,N)
        for item , pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)
    
    
def Popularity(train, test, result, N = 5000):
    item_popularity = dict() ##得到训练集中，所有的物品被消费的次数
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    
    ret = 0
    n = 0
    for user in test.keys():  ##拿测试集中的user进行推荐
        rank = GetRecommendation(result,user,N)  ##得到某个user的推荐列表
        for item,pui in rank:
            ret += math.log(1 + item_popularity[item])  ##user的推荐列表的物品的消费次数，然后取对数
            n += 1
    ret /= n * 1.0  ###在测试集上的用户，推荐的列表
    return ret