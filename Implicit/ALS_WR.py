# -*- coding: utf-8 -*-
"""
create on 2018-5-22 20:59:44
by：weifeng

"""
####ALS_WR模型####
import scipy as sp
import numpy as np
import pandas as pd
import time
import lightfm.evaluation
from lightfm import LightFM
from implicit.als import AlternatingLeastSquares
import sklearn.metrics
import math
import random
import operator
def generate_hot_item_list(plays, top=100):  ###根据测试集的数据生成热门列表？
    item_indexs, item_counts = np.unique(plays.col, return_counts=True) ###返回测试集合中所有的物品，和计数
    items_played_count = filter(lambda item_pair: item_pair[1] > 10, list(zip(item_indexs, item_counts))) ##出现次数超过10次以上的算热门
    return sorted(list(items_played_count), key=lambda i: i[1], reverse=True)[: top]
###如何实现加权采样
######测试pointwise类别
##加权采样有很多实现方法，最简单合理的是利用指数分布来实现。加权采样 的用途也很多，比如
###热门程度越高采样的几率就越大
def weighted_sampling(sequence, k):
    """
    parameters:
    sequence -- list-like [(item1, weight1), ...]
    k -- number of selected items
    return:
    list that selected.
    """
    weighted_list = []
    for elements in sequence:         ###random.expovariate(lamba)是一个随机值，期望是1/lamba,所以当前的值越大，这个值越小
        weighted_list.append((elements[0], random.expovariate(elements[1])))  ###random.expovariate(lambd) # 随机生成符合指数分布的随机数，lambd为指数分布的参数

    return sorted(weighted_list, key=lambda x: x[1])[:k]
##hot_items：测试集中用户的热门
##play:物品总共的集合
###uindex，用户的总共的集合
def generate_negative_samples(uindex, plays, hot_items, negative_count = 5):
    history = set(plays.getrow(uindex).indices)
    candidates = []
    for (item, weight) in hot_items:
        if item in history:          ###测试集热门物品中，用户没有消费过的物品作为
            continue
        candidates.append((item, weight))  ###candinate以测试集合中的热门列表，来看用户有哪些没有消费的物品，然后
    if negative_count > len(candidates):
        negative_count = len(candidates)
    return weighted_sampling(candidates, negative_count)

###评估point_wise类,计算AUC
def evaluate_point_wise_model(model, plays, test, num_test_users = -1):
    hot_items = generate_hot_item_list(test.tocoo())  ###生成热门列表，item:count且是排好序的
    user_indexes = range(plays.shape[0])
    aucs = []
    if num_test_users > 0:
        user_indexes = np.random.choice(user_indexes, num_test_users)  ##批量抽选多少个用户
    for uindex in user_indexes:
        positive_samples = test.tocsr().getrow(uindex).indices  ###测试数据集合上用户消费的物品的索引
        negative_samples = generate_negative_samples(uindex, plays.tocsr(), hot_items, len(positive_samples))##生成和正样本等量的负样本
        if len(negative_samples) == 0:
            continue
        negative_samples, weight = zip(*negative_samples)
        negative_samples = np.array(negative_samples)
        user_factor = model.user_factors[uindex].reshape((1, model.factors))
        user_samples = np.concatenate((positive_samples,  negative_samples), axis = 0).astype(np.int64)
        user_feedback = np.concatenate((np.full(len(positive_samples), 1),   ####因为并不是0,1这类东西，所以误差很大
                                        np.full(len(negative_samples), 0)), axis = 0)
        item_factors = model.item_factors[user_samples]
        scores = np.dot(user_factor, item_factors.transpose()).reshape(len(user_feedback))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(user_feedback, scores, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
    return np.array(aucs)

####训练point_wise类
def train_point_wise_model_and_evaluate(train, plays=None, test=None, factors=20,regularization=1, epochs=50, learning_rate=0.05,
                                        num_test_users=-1, eva=True):
    tic = time.time()
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=epochs, num_threads=-1)
    model.fit(train.transpose())
    toc = time.time()
    print("ALS training cost %.2f seconds" % (toc - tic))

    if eva:  ###为了测试这里的plays不能为空
        eva_test = evaluate_point_wise_model(model, plays, test, num_test_users)
        print("User auc mean = %.2f, std = %.2f (on testing dataset)" % (eva_test.mean(), eva_test.std()))
        auc = eva_test.mean()   ##在这个地方做一个网格搜索
    return model,auc


def ALS_WR_Evaluation(model,plays,train,test):
    user_ids = set(test.tocoo().row)  ##测试集合的用户
    items_all = set(train.tocoo().col)
    items_all_double = train.tocoo().col  ##用户没有去重的用户的集合

    pre_recall_hits = 0
    recall_all = 0
    precis_all = 0
    popular = 0
    item_popularity = dict()  ###这里物品的流行度是按照人头来算的
    recommendall = set()
    for item in items_all_double:
        if item not in item_popularity.keys():
            item_popularity[item] = 1
        else:
            item_popularity[item] += 1

    for user_id in user_ids:  ###用户在测试集上
        items = np.array(range(train.shape[1]))                     ##这个得分是排好序的，而且是用户没有消费过的物品
        recommed_item_score = model.recommend(user_id,plays,N=100)  ##返回的是listoftump[(item,score)]
        """                                                  
        filter the items the user has consumed. 
        """
        history = set(train.getrow(user_id).indices)  ###当前这个用户的训练集物品
        recommendations = []
        for item in recommed_item_score:
            if item[0] in history:
                continue
            recommendations.append(item)  ##找到排名前10的用户没有进行消费的物品进行推荐
            if len(recommendations) >= 10:
                break
        test_cur = set(test.getrow(user_id).indices)
        recommendationslast = []
        for rec in recommendations:
            recommendationslast.append(rec[0])
            recommendall.add(rec[0])
        for cur_item in test_cur:
            if cur_item in recommendationslast:  ##推荐的物品在用户后面消费的物品中
                pre_recall_hits += 1
                popular += math.log(1 + item_popularity[cur_item])
        recall_all += len(test_cur)
        precis_all += len(recommendationslast)
    print("准确率", pre_recall_hits / (precis_all * 1.0))
    print("召回率", pre_recall_hits / (recall_all * 1.0))
    print("覆盖率", len(recommendall) / len(items_all))
    print("多样性", popular / pre_recall_hits * 1.0)
    ###流行度，这个物品被多少人消费了

class ALS_Recommender(object):
    def __init__(self, models, plays=None, artists=None):  ##初始化
        self.models = models
        self.plays = plays
        self.artists = artists
        self.artistsDict = None
        if artists is not None:
            index, names = zip(*list(enumerate(self.artists)))
            self.artistsDict = dict(zip(names, index))
    def recommend(self, userid, modelname='bpr', top=10, with_history=True):  ##推荐
        recommend_list = self._recommend_with_als(userid, top)
        return self._output_more(userid, None, recommend_list, with_history)

    def similar_items(self, artist_name, top=10):    ###给定作家的名字看作家的相似度
        if artist_name not in self.artistsDict:
            return {}
        itemid = self.artistsDict[artist_name]
        model = self.models
        similar_items = model.similar_items(itemid, top)
        return self._output_more(None, itemid, similar_items, False)
    def _recommend_with_als(self, userid, top):     ###利用ALS算法进行推荐
        model = self.models
        return model.recommend(userid, self.plays, N=top)

    def _output_more(self, userid, itemid, item_list, with_history):
        userinfo = []
        output_iteminfo = []
        input_iteminfo = []
        if userid and with_history:
            userinfo = self._output_user_more_info(userid)
        if item_list:
            output_iteminfo = self._output_items_more_info(item_list)
        if itemid:
            input_iteminfo = self._output_items_more_info([(itemid, 1)])
        return {'user': userinfo, 'item': input_iteminfo, 'items': output_iteminfo}

    def _output_user_more_info(self, userid, sort=False, top=-1):
        history = self.artists[self.plays.getrow(userid).indices]
        playcount = self.plays.getrow(userid).data

        if not sort:
            return list(zip(history, playcount))[: top]
        else:
            return sorted(list(zip(history, playcount)), key=lambda item: item[1], reverse=True)[: top]

    def _output_items_more_info(self, items):
        itemids, scores = zip(*items)
        iteminfo = self.artists[list(itemids)]
        return list(zip(iteminfo, scores))



