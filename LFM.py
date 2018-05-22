# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:51:55 2018

@author: lanlandetian
"""

import random
import operator

allItemSet = set()
def InitAllItemSet(user_items):
    allItemSet.clear()
    for user, items in user_items.items():
        for i, r in items.items():
            allItemSet.add(i)

def InitItems_Pool(items):
    interacted_items = set(items.keys())
    items_pool = list(allItemSet - interacted_items)  ##用户没有消费的物品
#    items_pool = list(allItemSet)
    return items_pool
##选择某个用户的负样本
def RandSelectNegativeSample(items):
    ret = dict() ##某个用户消费的所有的key
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0,len(items) * 3):  ##这里的3代表了正负样本的比例
        items_pool = InitItems_Pool(items)  ##这一行代码，可以踢出去，放在for的后面
        item = items_pool[random.randint(0,len(items_pool) - 1 )]
        if item in ret:
            continue
        ret[item] = 0  ##以用户没有消费的物品作为负样本正负样本的比例，按照1:3来进行
        n += 1
        if n > len(items):  ##在这个地方，为了实现样本的均衡1:1
            break
    shufflelist = list(ret.keys())  ##做一个乱序操作
    random.shuffle(shufflelist)
    rett = {}
    for keys in shufflelist:
        rett[keys] = ret[keys]
    return rett

def Predict(user,item,P,Q):
    rate = 0
    for f,puf in P[user].items():
        qif = Q[item][f]
        rate += puf * qif
    return rate


def InitModel(user_items,F):
    P = dict()  ##user_items:用户，物品集合
    Q = dict()
    for user, items in user_items.items():
        P[user] = dict()
        for f in range(0,F):  ##用户的F个特征
            P[user][f] = random.random()  ##随机填充了一个【0....1)的数据
        for i,r in items.items():  ##物品的F个特征
            if i not in Q:
                Q[i] = dict()
                for f in range(0,F):
                    Q[i][f] = random.random()  ##填充的是随机数
    return P,Q



def LatentFactorModel(user_items, F,T, alpha, lamb):
    InitAllItemSet(user_items)  ##得到所有的物品集合
    [P,Q] = InitModel(user_items, F)
    for step in range(0,T):
        for user, items in user_items.items():
            samples = RandSelectNegativeSample(items)
            for item, rui in samples.items():
                eui = rui - Predict(user, item, P,Q)
                for f in range(0,F):                        ##更新隐含的特征值
                    P[user][f] += alpha * (eui * Q[item][f] - \
                        lamb * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - \
                        lamb * Q[item][f])
        alpha *= 0.9
    return P,Q
        
def Recommend(user,train,P,Q):
    rank = dict()
    if user not in train.keys():
        return rank
    interacted_items = train[user] ##训练集中user消费的物品
    for i in Q:
        if i in interacted_items.keys():  ##过滤掉用户在训练集上消费的物品
            continue
        rank.setdefault(i,0)
        for f,qif in Q[i].items():
            puf = P[user][f]
            rank[i] += puf * qif
    return rank         ##得到其余物品的得分
    
def Recommendation(users, train,P,Q):
    result = dict()
    for user in users:
        rank = Recommend(user,train,P,Q)
        R = sorted(rank.items(), key = operator.itemgetter(1), \
                   reverse = True)
        result[user] = R
    return result