import math
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def dcg_k(actual, predicted, topk):
    """
    获取预测值的dcg评分

    :param actual: 真实的用户选择
    :param predicted: 预测的用户选择
    :param topk: 最高需要判断多长的连续的子序列
    :return: 返回每个用户的dcg得分
    """

    k = min(topk, len(actual))

    dcgs = []
    # 将预测序列和真实序列转成numpy格式
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()

    for user_id in range(len(actual)):
        # 分别计算每个用户的dcg_k值
        value = []
        
        for i in predicted[user_id]:
            try:
                value += [topk - int(np.argwhere(actual[user_id] == i))]
            except:
                value += [0]

        dcg_k = sum([value[j] / math.log(j + 2, 2) for j in range(k)])

        if dcg_k == 0:
            dcg_k = 1e-5
        dcgs.append(dcg_k)

    return dcgs


def bleu(hyps, refs):
    bleu_4 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1./3, 1./3, 1./3, 0])
        except:
            score = 0
        bleu_4.append(score)
    bleu_4 = np.average(bleu_4)
    return bleu_4


def bleu_each(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    hyps=hyps.cpu().numpy()
    refs=refs.cpu().numpy()
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    return bleu_4

def precision_at_k(actual, predicted, topk,item_i):
    sum_precision = 0.0
    user = 0
    num_users = len(predicted)
    for i in range(num_users):
        if actual[i][item_i]>0:
            user +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                sum_precision += 1
        else:
            continue
    #print(user)
    return sum_precision / user


def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    user = 0
    for user_id in range(len(actual)):
        if actual[user_id][item_i] > 0:
            user +=1
            dcg_k = sum([int(predicted[user_id][j] in [actual[user_id][item_i]]) / math.log(j+2, 2) for j in range(k)])
            res += dcg_k
        else:
            continue
    #print(user)
    return res/user