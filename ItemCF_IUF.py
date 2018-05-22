# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:58:20 2018

@author: lanlandetian
"""

import math
import operator

def ItemSimilarity(train):   ##{user:{item1:1,item2:1....}
    #calculate co-rated users between items
    C = dict()
    N = dict()    ###{item:count},表示的是item被count用户消费
    for u, items in train.items():  ##这里的u代表的是user,item代表的是user对item的行为
        for i in items.keys():            ####这一个用户
            N.setdefault(i,0)   ###将第一个位置，设置为{i:0}，为下一步进行累计提前做好
            N[i] += 1           ###setdefault 如果键存在的时候，返回键值。如果不存在设置为默认的值
            C.setdefault(i,{})
            for j in items.keys():
                if i == j:
                    continue
                C[i].setdefault(j,0)  ##设置c[i][j]= 0
                C[i][j] += 1 / math.log(1+len(items) * 1.0)  ##c[i][j]代表的是中间的结果，表示的是同一个用户的
                                                             ##对用户的同一个物品的数量进行惩罚
                                                             ##i，j同属于一个用户
    #calculate finial similarity matrix W
    W = C.copy()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W
    
def Recommend(user_id,train, W,K = 3):
    rank = dict()
    if user_id not in train.keys():
        return rank
    ru = train[user_id]  ###在训练集中找到了user曾经作用过的物品
    for i,pi in ru.items():  ##i是训练集物品编号
        for j,wij in sorted(W[i].items(), \
                           key = operator.itemgetter(1), reverse = True)[0:K]:  ##找到最相似的K个
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j] += pi *wij
    return rank
    
#class Node:
#    def __init__(self):
#        self.weight = 0
#        self.reason = dict()
#
#def Recommend(user_id,train, W,K=5):
#    rank = dict()
#    ru = train[user_id]
#    for i,pi in ru.items():
#        for j,wij in sorted(W[i].items(), \
#                           key = operator.itemgetter(1), reverse = True)[0:K]:
#            if j in ru:
#                continue
#            if j not in rank:
#                rank[j] = Node()
#            rank[j].reason.setdefault(i,0)
#            rank[j].weight += pi *wij
#            rank[j].reason[i] = pi * wij
#    return rank
    
def Recommendation(users, train, W, K = 3):
    result = dict()
    for user in users:   ##这里的user是test的user
        rank = Recommend(user,train,W,K) ##用户在训练集中的每一个物品都找，不属于用户的，最相似的K个
        R = sorted(rank.items(), key = operator.itemgetter(1), \
                   reverse = True)
        result[user] = R
    return result