import torch.nn as nn
from model.dynamicGRU import DynamicGRU
import torch
import torch.nn.functional as F
from eval_metrics import *


class KERL(nn.Module):
    def __init__(self, num_user, num_items, model_args, device, kg_map):
        super(KERL, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 10

        L = self.args.L
        dims = self.args.d
        predict_T = self.args.T

        self.kg_map = kg_map
        # 对商品的id做embedding的层，这里还是无关知识的，只是用于将id做一个embedding
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.DP = nn.Dropout(0.5)
        self.enc = DynamicGRU(input_dim=dims, output_dim=dims, bidirectional=False)
        # 这个mlp的输入就是gru的输出的隐状态和池化后的喜好表示和预测未来的喜好表示
        self.mlp_history = nn.Linear(50, 50)

        # 这个mlp的输入就是前面三种表示的拼接，然后通过fc输出一个可能选取的商品的概率
        self.mlp = nn.Linear(dims + 50 * 2, dims * 2)
        self.fc = nn.Linear(dims * 2, num_items)
        self.softmax=nn.Softmax(1)


        self.BN = nn.BatchNorm1d(50, affine=False)
        # 计算余弦相似度的层
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, batch_sequences, train_len):
        """
        :param batch_sequences: 多个batch的用户购买序列(b,l,f)
        :param train_len: 每个序列中的batch的个数
        :return:
        """
        probs = []
        # 对输入的序列进行embedding
        input = self.item_embeddings(batch_sequences)
        # 进GRU获取到序列级的知识表示
        out_enc, h = self.enc(input, train_len)

        # 对商品的嵌入向量做一个批量归一化
        kg_map = self.BN(self.kg_map)
        kg_map = kg_map.detach()
        batch_kg = self.get_kg(batch_sequences, train_len, kg_map)

        mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

    def get_kg(self, batch_sequence, train_len, kg_map):
        """
        获取用户购买商品的嵌入的平均池化以获取当前的喜好表示

        :param batch_sequence: 用户的一系列购买记录
        :param train_len: 每个序列包含的batch数量
        :param kg_map: 商品的嵌入向量表
        :return: 平均池化之后的batch sequence，会降一维吧
        """
        batch_kg = []
        # 下面就是做一个平均池化，获取当前的喜好
        for i, seq in enumerate(batch_sequence):
            # sequence中是内部id也就是0，1，2，3，因此需要用kg_map将他映射成嵌入向量

            seq_kg = kg_map[seq]
            seq_kg_avg = torch.sum(seq_kg, dim=0)
            seq_kg_avg = torch.div(seq_kg_avg, train_len[i])
            batch_kg.append(seq_kg_avg)

        # 将池化之后的数据再次拼接起来
        batch_kg = torch.stack(batch_kg)
        return batch_kg

    def RL_train(self, batch_sequences, items_to_predict, pred_one_hot, train_len, target_len):
        """
        RL训练的一个过程

        :param batch_sequences: 一个batch中的选到的用户的选择序列
        :param items_to_predict: 需要预测的真实序列
        :param pred_one_hot: 真实序列编码的类似one-hot信息
        :param train_len: batch_sequences中的每个序列的长度
        :param target_len: 真实序列中每个序列的长度
        :return: 预测的第一个的带真实标签的概率分布,第一个的不带真实标签的概率分布,第一个的sample集合,总奖励,按照seq奖励高低排序的kg奖励(包含三次)
        """

        probs = []  # 加上了真实标签信息的概率
        probs_origin = []  # 模型原始的mlp输出的概率
        each_sample = []
        Rewards = []

        # 对序列id做一个embedding,无关知识,形状应该是[[[...],[...],...],...],0维的长度是当前batch的用户数量
        input = self.item_embeddings(batch_sequences)
        # 使用gru编码序列层面的表示
        out_enc, h = self.enc(input, train_len)

        # 根据知识图获取当前的喜好知识表示
        kg_map = self.BN(self.kg_map)
        batch_kg = self.get_kg(batch_sequences, train_len, kg_map)

        # 根据当前喜好获取未来的喜好，并且做一个cat
        mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)

        # 再用一个mlp对上面的结果做一个映射，最终fc的输出的维度是总商品数量
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        # 做一个softmax，获取可能的商品概率
        out_distribution = F.softmax(out_fc, dim=1)
        # out_distribution = self.softmax(out_fc)
        probs_origin.append(out_distribution)
        out_distribution =out_distribution* 0.8
        # 将真实的标签的信息添加到预测的概率上
        out_distribution = torch.add(out_distribution, pred_one_hot)
        probs.append(out_distribution)

        # 设置文中的 pi(a|s)
        """
        这里是不是有点问题，Categorical里面的第一个参数是概率 但是这里的out_distribution
        是加上了pred_one_hot 的，就是说它的总和一定大于1，就不是个概率了
        
        哦这里没问题，函数内部会对它进行再一次归一化的，这里看成传入的相对概率
        """
        m = torch.distributions.categorical.Categorical(out_distribution)
        # 从这个分布中随机抽取一个索引（按照分布来的）
        sample1 = m.sample()
        each_sample.append(sample1)

        # 对于该action我们获取奖励
        Reward, dist_sort = self.generateReward(sample1, self.args.T - 1, 3, items_to_predict,
                                                pred_one_hot, h, batch_kg, kg_map, target_len)
        Rewards.append(Reward)

        probs = torch.stack(probs, dim=1)
        probs_origin = torch.stack(probs_origin, dim=1)

        return probs, probs_origin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1), dist_sort

    def generateReward(self, sample1, path_len, path_num, items_to_predict,
                       pred_ont_hot, h_origin, batch_kg, kg_map, target_len):
        """

        :param sample1: action，是一个商品的id（内部id 整数形式的）
        :param path_len: target_len 减去1，因为已经有了一个预测值了 就是传进来的sample1，所以少预测一个
        :param path_num: 预测几次,取总和的平均值作为代表
        :param items_to_predict: 真实标签
        :param pred_ont_hot: 真实标签的独热编码（对应位置的值是0.2）
        :param h_origin: gru输出的h
        :param batch_kg: 当前的知识喜好表示
        :param kg_map: 知识图
        :param target_len: 每一条标签的长度
        :return: 最终的奖励值,按照seq奖励的高低排序的kg奖励
        """
        # 获取未来的喜好表示
        history_kg = self.mlp_history(batch_kg)

        Reward = []
        dist = []
        dist_replay = []

        for paths in range(path_num):
            h = h_origin
            # 保存最终所推荐的那几个商品的id
            indexes = []
            indexes.append(sample1)

            # 和h相配合放入到GRU中进行预测下一个推荐用的
            dec_inp_index = sample1
            # 将内部id做一个embedding。与知识无关
            dec_inp = self.item_embeddings(dec_inp_index)
            dec_inp = dec_inp.unsqueeze(1)

            # 获取真实的知识图的平均池化
            """
            下面这一句没看懂，从传过来的参数来看，path_len=args.T-1,那么后面的参数就是0,等于没有啊
            ground_kg=self.get_kg(items_to_predict[:,self.args.T-path_len-1:], target_len, kg_map)
            完全可以改掉
            """
            ground_kg = self.get_kg(items_to_predict, target_len, kg_map)

            for i in range(path_len):
                # 用预测的第一时刻放入gru以预测之后时刻
                out_enc, h = self.enc(dec_inp, h, one=True)

                # 获取结合了三种表示的状态表示
                mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
                # 获取三种状态表示的
                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                # out_distribution = self.softmax(out_fc)
                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution =out_distribution* 0.8
                """
                这里也有点小问题,为什么还要加上真实标签的概率呢?
                (如果效果不好的话可以不加试一下)
                """
                out_distribution = torch.add(out_distribution, pred_ont_hot)

                # 往后预测一个
                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()

                dec_inp = self.item_embeddings(sample2)
                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)
            # 此时indexes中存储的是config.T-1个预测的商品id
            indexes = torch.stack(indexes, dim=1)
            # 获取这三个商品对应的当前喜好的知识表示
            episode_kg = self.get_kg(indexes, torch.Tensor([path_len + 1] * len(indexes)), kg_map)

            # 获取我们预测的商品的平均池化，与ground_truth求一个距离
            dist.append(self.cos(episode_kg, ground_kg))
            dist_replay.append(self.cos(episode_kg, history_kg))

            # 根据两个不同的预测结果获取奖励得分
            """
            这里和上面一样，不需要这么做吧
            
            Reward.append(dcg_k(items_to_predict[:,self.args.T-path_len-1:],indexes,path_len+1))
            
            此外这里用的是dcg方法,但是文章中使用的是bleu方法,我们将其改为bleu方法
            
            Reward.append(dcg_k(items_to_predict, indexes, path_len + 1))
            """
            Reward.append(bleu_each(items_to_predict,indexes))

        Reward = torch.FloatTensor(Reward).to(self.device)
        # 将多次的预测取平均
        dist = torch.stack(dist, dim=0)
        dist = torch.mean(dist, dim=0)

        dist_replay = torch.stack(dist_replay, dim=0)
        dist_sort = self.compare_kgReward(Reward, dist_replay)

        # 多次的平均奖励值
        Reward = torch.mean(Reward)
        Reward = Reward + self.lamda * dist
        dist_sort = dist_sort.detach()
        return Reward, dist_sort

    def compare_kgReward(self, reward, dist):
        """
        将dist按照reward的每列最大值的索引进行重新排个序

        :param reward: 奖励值
        :param dist: 每个预测商品和target的余弦相似度
        :return: 排好序的dist
        """
        logit_reward, indice = reward.sort(dim=0)
        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort
