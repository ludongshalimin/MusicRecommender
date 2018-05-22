####听歌的次数作为隐式反馈
####根据隐式反馈构建用户的推荐系统
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


# small_data = pd.read_csv(filepath_or_buffer='./small_data.csv',encoding="UTF-8")
# small_data['user'] = small_data['user'].astype("category")
# small_data['artist'] = small_data['artist'].astype("category")
#                                                                                     ###共计100000条数据
# plays = sp.sparse.coo_matrix((small_data['plays'].astype(float),                         ###943 * 1682
#         (small_data['user'].cat.codes, small_data['artist'].cat.codes)), dtype = np.double)   ####这个矩阵构造的就是用户的矩阵

##如果使用的原来的数据
small_data = pd.read_csv(filepath_or_buffer='./u.data',encoding="UTF-8",
                         sep='\t', header=None,
                         usecols =[0,1,2],
                         names=['user', 'artist', 'plays']
                         )
small_data['user'] = small_data['user'].astype("category")
small_data['artist'] = small_data['artist'].astype("category")
                                                                                    ###共计100000条数据
plays = sp.sparse.coo_matrix((small_data['plays'].astype(float),                         ###943 * 1682
        (small_data['user'].cat.codes, small_data['artist'].cat.codes)), dtype = np.double)   ####这个矩阵构造的就是用户的矩阵
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

train_mat, test_mat = split_train_test(plays.tocsr())
model = train_pair_wise_model_and_evaluate(train_mat, test = test_mat)

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
BPR_Evaluation(model,train_mat,test_mat)







####对ALS进行测试


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
    hot_items = generate_hot_item_list(test_mat.tocoo())  ###生成热门列表，item:count且是排好序的
    user_indexes = range(plays.shape[0])
    aucs = []
    if num_test_users > 0:
        user_indexes = np.random.choice(user_indexes, num_test_users)  ##批量抽选多少个用户
    for uindex in user_indexes:
        positive_samples = test_mat.tocsr().getrow(uindex).indices  ###测试数据集合上用户消费的物品的索引
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
def train_point_wise_model_and_evaluate(train, plays=None, test=None, factors=20, epochs=50, learning_rate=0.05,
                                        num_test_users=-1, eva=True):
    tic = time.time()
    model = AlternatingLeastSquares(factors=factors, regularization=1, iterations=epochs, num_threads=-1)
    model.fit(train.transpose())
    toc = time.time()
    print("ALS training cost %.2f seconds" % (toc - tic))

    if eva:  ###为了测试这里的plays不能为空
        eva_test = evaluate_point_wise_model(model, plays, test, num_test_users)
        print("User auc mean = %.2f, std = %.2f (on testing dataset)" % (eva_test.mean(), eva_test.std()))
    return model

model2 = train_point_wise_model_and_evaluate(train_mat, plays, test_mat,factors = 100, epochs = 10, num_test_users = 100)
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
ALS_WR_Evaluation(model2,train_mat.tocsr(),train_mat,test_mat)








model1 = train_pair_wise_model_and_evaluate(train_mat, test = None, factors = 100, epochs = 10, eva = False)
model2 = train_point_wise_model_and_evaluate(train_mat, factors = 100, epochs = 10, eva = False)

class Recommender(object):
    def __init__(self, models={}, plays=None, artists=None):  ##初始化
        self.models = models
        self.plays = plays
        self.artists = artists
        self.artistsDict = None
        if artists is not None:
            index, names = zip(*list(enumerate(self.artists)))
            self.artistsDict = dict(zip(names, index))
    def recommend(self, userid, modelname='bpr', top=10, with_history=True):  ##推荐
        if modelname not in self.models:
            return []
        recommend_list = []
        if modelname == 'bpr':
            recommend_list = self._recommend_with_bpr(userid, top)
        elif modelname == 'als':
            recommend_list = self._recommend_with_als(userid, top)

        return self._output_more(userid, None, recommend_list, with_history)

    def similar_items(self, artist_name, top=10):    ###给定作家的名字看作家的相似度
        if artist_name not in self.artistsDict:
            return {}
        itemid = self.artistsDict[artist_name]
        model = self.models['als']
        similar_items = model.similar_items(itemid, top)
        return self._output_more(None, itemid, similar_items, False)

    def _recommend_with_bpr(self, userid, top):     ###利用BPR算法进行推荐
        """
        compute recommendation for user
        """
        model = self.models['bpr']
        items = np.array(range(plays.shape[1]))
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

    def _recommend_with_als(self, userid, top):     ###利用ALS算法进行推荐
        model = self.models['als']
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


# 25 in list(small_data['user'].cat.codes)

##给cat.codes为25号的人推荐音乐
recommender = Recommender({'bpr': model1, 'als':model2}, plays.tocsr(), small_data.artist.cat.categories)
recommendation1 = recommender.recommend(25, modelname='bpr', top = 20, with_history = False)
recommendation2 = recommender.recommend(25, modelname='als', top = 20, with_history = False)
userhistory = recommender._output_user_more_info(25, sort = True)
print("user recommend with bpr",recommendation1)
print("user recommend with als",recommendation2)
print("user history with sort",userhistory)
