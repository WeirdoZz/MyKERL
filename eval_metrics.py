import math

import numpy as np



def dcg_k(actual,predicted,topk):
    """
    获取预测值的dcg评分
    :param actual: 真实的用户选择
    :param predicted: 预测的用户选择
    :param topk: 选取前k个预测值表示整体
    :return: 返回每个用户的dcg得分
    """
    k=min(topk,len(actual))
    dcgs=[]
    actual=actual.cpu().numpy()
    predicted=predicted.cpu().numpy()

    for user_id in range(len(actual)):
        # 分别计算每个用户的dcg_k值
        value=[]
        for i in predicted[user_id]:
            try:
                value+=[topk-int(np.argwhere(actual[user_id]==i))]
            except:
                value+=[0]

        dcg_k=sum([value[j]/math.log(j+2,2) for j in range(k)])

        if dcg_k==0:
            dcg_k=1e-5
        dcgs.append(dcg_k)

    return dcgs