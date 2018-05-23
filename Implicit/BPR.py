# -*- coding: utf-8 -*-
"""
create on 2018-5-22 20:59:44
by：weifeng

"""
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
####使用BPR模型######
def train_pair_wise_model_and_evaluate(train, test=None, factors=50, epochs=10, learning_rate=0.05, loss='bpr',
                                       eva=True):
    tic = time.time()
    model = LightFM(no_components=factors, learning_rate=learning_rate, loss=loss)
    model.fit(train, epochs=epochs, num_threads=2)
    toc = time.time()
    print("LightFM training cost %.2f seconds" % (toc - tic))

    if test is not None and eva:
        train_precision = lightfm.evaluation.precision_at_k(model, train, k=10,num_threads=-1)
        train_recall = lightfm.evaluation.recall_at_k(model, train, k=10,num_threads=-1)
        eva_train = lightfm.evaluation.auc_score(model, train, num_threads=-1)

        print("User precision mean = %.2f on train" % (train_precision.mean()))
        print("User recall mean = %.2f on train" % (train_recall.mean()))
        print("User auc mean = %.2f, std = %.2f (on training dataset)" % (eva_train.mean(), eva_train.std()))

        test_precision = lightfm.evaluation.precision_at_k(model, test, k=10,num_threads=-1)
        test_recall = lightfm.evaluation.recall_at_k(model, test, k=10,num_threads=-1)
        eva_test = lightfm.evaluation.auc_score(model, test, num_threads=-1)
        print("User precision mean = %.2f on test" % (test_precision.mean()))
        print("User recall mean = %.2f on test" % (test_recall.mean()))
        print("User auc mean = %.2f, std = %.2f (on testing dataset)" % (eva_test.mean(), eva_test.std()))
    return model
###根据推荐结果自己手动实现覆盖率和多样性？
def BPR_Evaluation(model,train,test):
    user_ids = set(test.tocoo().row)    ##测试集合的用户
    items_all = set(train.tocoo().col)
    items_all_double = train.tocoo().col  ##用户没有去重的用户的集合

    pre_recall_hits = 0
    recall_all = 0
    precis_all = 0
    popular = 0
    item_popularity = dict()      ###这里物品的流行度是按照人头来算的
    recommendall = set()  ###这里的recommendall是所有测试集合用户推荐的物品
    for item in items_all_double:
        if item not in item_popularity.keys():
            item_popularity[item]=1
        else:
            item_popularity[item] += 1

    for user_id in user_ids:  ###用户在测试集上
        items = np.array(range(train.shape[1]))
        scores = model.predict(user_id, items)
        sorted_items = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        """
        filter the items the user has consumed. 
        """
        history = set(train.getrow(user_id).indices)  ###当前这个用户的训练集物品
        recommendations = []
        for item in sorted_items:
            if item[0] in history:
                continue
            recommendations.append(item)    ##找到排名前10的用户没有进行消费的物品进行推荐
            if len(recommendations) >= 10:
                break
        test_cur = set(test.getrow(user_id).indices)
        recommendationslast =[]

        for rec in recommendations:
            recommendationslast.append(rec[0])
            recommendall.add(rec[0])
        for cur_item in test_cur:
            if cur_item in recommendationslast:    ##推荐的物品在用户后面消费的物品中
                pre_recall_hits += 1
                popular += math.log(1 + item_popularity[cur_item])
        recall_all += len(test_cur)
        precis_all += len(recommendationslast)
    print("准确率",pre_recall_hits/(precis_all*1.0))
    print("召回率",pre_recall_hits/(recall_all*1.0))
    print("覆盖率",len(recommendall)/len(items_all))
    print("多样性",popular /pre_recall_hits *1.0)
    ###流行度，这个物品被多少人消费了

##实现BPR推荐的接口
class BPR_Recommender(object):
    def __init__(self, models, plays=None, artists=None):  ##初始化
        self.models = models
        self.plays = plays
        self.artists = artists
        self.artistsDict = None
        if artists is not None:
            index, names = zip(*list(enumerate(self.artists)))
            self.artistsDict = dict(zip(names, index))
    def recommend(self, userid, top=10, with_history=True):  ##推荐
        recommend_list = self._recommend_with_bpr(userid, top)
        return self._output_more(userid, None, recommend_list, with_history)

    def similar_items(self, artist_name, top=10):  ### 对于BPR模型来说没有相似的物品的一说
        pass
    def _recommend_with_bpr(self, userid, top):  ###利用BPR算法进行推荐
        """
        compute recommendation for user
        """
        model = self.models
        items = np.array(range(self.plays.shape[1]))
        scores = model.predict(userid, items)
        sorted_items = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        """
        filter the items the user has consumed. 
        """
        history = set(self.plays.getrow(userid).indices)
        recommendations = []
        for item in sorted_items:
            if item[0] in history:
                continue
            recommendations.append(item)
            if len(recommendations) >= top:
                break
        return recommendations
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

    def _output_user_more_info(self, userid, sort=False, top=-1):  ###输出用户更多的信息
        history = self.artists[self.plays.getrow(userid).indices]
        playcount = self.plays.getrow(userid).data
        if not sort:
            return list(zip(history, playcount))[: top]
        else:
            return sorted(list(zip(history, playcount)), key=lambda item: item[1], reverse=True)[: top]
    def _output_items_more_info(self, items):     ####输出物品更多的信息
        itemids, scores = zip(*items)
        iteminfo = self.artists[list(itemids)]
        return list(zip(iteminfo, scores))

