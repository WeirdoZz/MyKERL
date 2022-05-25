import argparse
import numpy as np
import torch
from eval_metrics import precision_at_k, ndcg_k
from model.KERL import KERL
from data import Amazon
from interactions import Interactions
import datetime
import logging
from time import time
import random
import math
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def generate_testsample(test_set,item_num):
    """
    生成测试序列

    :param test_set: 测试序列
    :param item_num: 商品总数
    :return: 返回一个随机的101长度的测试序列，共num_user个
    """

    all_sample=[]
    for eachset in test_set:
        testsample=[]
        for i in range(1):
            onesample=[]
            # 将测试序列的第一个商品放到onesample中
            onesample+=[eachset[i]]
            # 从所有商品的id中去掉第一个商品的id
            other=list(range(1,item_num))
            other.remove(eachset[i])
            # 再随机抽取100个负样本加到后面去
            neg=random.sample(other,100)
            onesample+=neg
            testsample.append(onesample)
        testsample = np.stack(testsample)
        all_sample.append(testsample)
    all_sample = np.stack(all_sample)
    return all_sample


def evaluation_kerl(kerl,train,test_set):
    num_users = train.num_users
    num_items = train.num_items

    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes=np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    test_sequences = train.valid_sequences.sequences
    test_len = train.valid_sequences.length
    # 生成测试样本
    all_sample = generate_testsample(test_set, num_items)

    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size

        if batch == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index=user_indexes[start:end]
        batch_valid_sequences=test_sequences[batch_user_index]
        batch_valid_sequences = np.atleast_2d(batch_valid_sequences)
        batch_valid_len = test_len[batch_user_index]

        batch_valid_len=torch.from_numpy(batch_valid_len).type(torch.LongTensor).to(device)
        batch_valid_sequences=torch.from_numpy(batch_valid_sequences).type(torch.LongTensor).to(device)

        prediction_score=kerl(batch_valid_sequences,batch_valid_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()

        if batch==0:
            pred_list=rating_pred
        else:
            pred_list = np.append(pred_list, rating_pred, axis=0)

    all_top10=[]
    for i in range(1):
        oneloc_top10=[]
        user_index=0
        for each_policy,each_s in zip(pred_list[:,i,:],all_sample[:,i,:]):
            # 取出预测的对应前面随机采样的商品id的概率并且取负数
            each_sample=-each_policy[each_s]
            # 获取最大的10个商品的索引
            top10index = np.argsort(each_sample)[:10]
            # 从商品中获取这10个商品
            top10item = each_s[top10index]
            oneloc_top10.append(top10item)
        oneloc_top10=np.stack(oneloc_top10)
        all_top10.append(oneloc_top10)
        pred_list = all_top10
    all_top10 = np.stack(all_top10, axis=1)
    pred_list = all_top10

    precision, ndcg = [], []
    k = 10
    for i in range(1):
        pred = pred_list[:, i, :]
        precision.append(precision_at_k(test_set, pred, k, i))
        ndcg.append(ndcg_k(test_set, pred, k, i))

    # save results
    # def save_obj(obj, name):
    #     with open(name + '.pkl', 'wb') as f:
    #         pickle.dump(obj, f)
    # str_name = "./result/LFM"+str(precision[0])
    # save_obj(pred_list, str_name)
    return precision, ndcg


def train_kerl(train_data,test_data,config,kg_map):
    # 获取用户的数量和商品的总数量
    num_users=train_data.num_users
    num_items=train_data.num_items

    # 获取训练集的序列的相关数据
    sequences_np=train_data.sequences.sequences
    targets_np=train_data.sequences.targets
    user_np=train_data.sequences.user_ids
    trainlen_np = train_data.sequences.length
    tarlen_np = train_data.sequences.tarlen

    n_train=len(sequences_np)
    logger.info("训练集的数据条数:{}".format(n_train))
    # 知识表示转换成tensor表示，放到gpu上
    kg_map=torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad=False

    seq_model=KERL(num_users,num_items,config,device,kg_map).to(device)
    optimizer=torch.optim.Adam(seq_model.parameters(),lr=config.learning_rate,weight_decay=config.l2)

    # 两个计算损失的地方,最终的损失是这两个损失按照权重相加
    lamda=5
    CEloss = torch.nn.CrossEntropyLoss()
    margin=0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin)

    record_indexes = np.arange(n_train) # 训练集的索引
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    # 控制提前结束训练的参数
    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False

    for epoch in range(config.n_iter):
        t1=time()
        loss=0
        seq_model.train()

        # 将索引打乱用于之后的随机分batch
        np.random.shuffle(record_indexes)

        epoch_reward=0.0  # 当前周期的奖励
        epoch_loss=0.0  # 当前周期的损失

        for batch in range(num_batches):
            start=batch*batch_size
            end=start+batch_size

            if batch==num_batches-1:
                if start<n_train:
                    end=n_train
                else:
                    break

            batch_record_index=record_indexes[start:end] # 当前batch的对应的数据的索引(mask)

            # 获取当前batch对应的数据
            batch_users=user_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            trainlen = trainlen_np[batch_record_index]
            tarlen = tarlen_np[batch_record_index]

            # 全部转成tensor格式
            tarlen = torch.from_numpy(tarlen).type(torch.LongTensor).to(device)
            trainlen = torch.from_numpy(trainlen).type(torch.LongTensor).to(device)
            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)

            # 真实标签
            items_to_predict = batch_targets

            # 这个if是必定满足的
            if epoch>=0:
                # 构建一个真实标签的类似one_hot的编码
                pred_one_hot = np.zeros((len(batch_users), num_items))
                # 获取当前batch的标签(本可以使用batch_targets的,但是上面已经转成tensor了,为了不影响后续,就重新获取一次)
                batch_tar = targets_np[batch_record_index]
                # 给每一个真实标签加上一点值
                for i,tar in enumerate(batch_tar):
                    """
                    这里原文代码是这样的,初一了一个target长度,但是如果是按照真实:预测 2:8的概率的话没必要用这个除以吧
                    
                    pred_one_hot[i][tar]=0.2/config.T
                    """
                    pred_one_hot[i][tar]=0.2
                pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

                prediction_score, origin, batch_targets, Reward, dist_sort = seq_model.RL_train(batch_sequences,
                                                                                              items_to_predict,
                                                                                              pred_one_hot, trainlen,
                                                                                              tarlen)
                target=torch.ones((len(prediction_score))).unsqueeze(1).to(device)

                # 最小的seq奖励对应的kg奖励和最大的seq奖励对应的kg奖励
                min_reward=dist_sort[0,:].unsqueeze(1)
                max_reward = dist_sort[-1, :].unsqueeze(1)
                # 求一个margin ranking 损失
                mrloss = MRLoss(max_reward, min_reward, target)

                """
                下面这里不是展平的意思，因为前面stack的时候多了一维，所以这里的展平相当于是一个squeeze操作
                """
                # 讲预测概率平铺开
                origin=origin.view(prediction_score.shape[0]*prediction_score.shape[1],-1)
                # 获取第一个选取的推荐商品，平铺开
                target=batch_targets.view(batch_targets.shape[0]*batch_targets.shape[1])
                # 讲对应的奖励平铺开，奖励的维度应该和target的维度是一样的
                reward = Reward.view(Reward.shape[0] * Reward.shape[1]).to(device)

                # 将所有用户的选择应用到每个用户上，得到一个2048*2048的张量
                prob=torch.index_select(origin,1,target)
                # 取其对角就是推荐给改用户的最大概率的商品的概率
                prob=torch.diagonal(prob,0)
                # 获取rl损失
                RLloss=-torch.mean(torch.mul(reward,torch.log(prob)))
                loss = RLloss + lamda * mrloss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss/=batch
        t2=time()
        output_str = "训练周期 %d [%.1f s]  loss=%.4f" % (epoch + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        if (epoch+1)>1:
            seq_model.eval()
            precision,ndcg=evaluation_kerl(seq_model, train_data, test_data)
            logger.info('精度'.join(str(e) for e in precision))
            logger.info('ndcg得分'.join(str(e) for e in ndcg))
            logger.info("验证时间:{}".format(time() - t2))
            cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                    break
    logger.info("\n")
    logger.info("\n")


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    """
    L:最大的序列长度
    T:事件的长度
    """

    parser.add_argument('--L', type=int, default=6)
    parser.add_argument('--T', type=int, default=2)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)

    config=parser.parse_args()

    dataset=Amazon.Beauty()
    train_set,test_set,num_users,num_items,kg_map=dataset.generate_dataset()

    """
    原代码给了这一段，但是这个max_len实际上用不到
    
    maxlen=0
    # 由于用户购买力不同，训练集的每组数据的条数不定相同
    for inter in train_set:
        if len(inter)>maxlen:
            maxlen=len(inter)        
    """

    train_data=Interactions(train_set, num_users, num_items)
    # 下面一步执行完之后,train_data中存在两个重要参数，
    # sequences:包含所有的训练部分的子序列和相对应的用户id以及序列真实长度（因为有些序列是被填充过了的），以及对应的target相关信息
    # valid_sequences:包含所有的验证部分的子序列基本和上面这个sequences相同
    train_data.to_newsequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)
    with torch.autograd.set_detect_anomaly(True):
        train_kerl(train_data, test_set, config, kg_map)