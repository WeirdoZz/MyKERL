import torch.nn as nn
from DynamicGRU import DynamicGRU
import torch
import torch.nn.functional as F
from eval_metrics import *

class KERL(nn.Module):
    def __init__(self,num_user,num_items,model_args,device,kg_map):
        super(KERL, self).__init__()

        self.args=model_args
        self.device=device
        self.lamda=10

        L=self.args.L
        dims=self.args.d
        predict_T=self.args.T

        self.kg_map=kg_map
        self.item_embeddings=nn.Embedding(num_items,dims).to(device)
        self.DP=nn.Dropout(0.5)
        self.enc=DynamicGRU(input_dim=dims,output_dim=dims,bidirectional=False)
        # 这个mlp的输入就是gru的输出的隐状态和池化后的喜好表示和预测未来的喜好表示
        self.mlp=nn.Linear(dims+50*2,dims*2)
        self.fc=nn.Linear(dims*2,num_items)
        self.mlp_history=nn.Linear(50,50)
        self.BN=nn.BatchNorm1d(50,affine=False)
        self.cos=nn.CosineSimilarity(dim=1,eps=1e-6)

    def forward(self,batch_sequences,train_len):
        """
        :param batch_sequences: 多个batch的用户购买序列
        :param train_len: 每个序列中的batch的个数
        :return:
        """
        probs=[]
        # 对输入的序列进行embedding
        input=self.item_embeddings(batch_sequences)
        # 进GRU获取到序列级的知识表示
        out_enc,h=self.enc(input,train_len)

        # 对商品的嵌入向量做一个批量归一化
        kg_map=self.BN(self.kg_map)
        kg_map=kg_map.detach()
        batch_kg=self.get_kg(batch_sequences,train_len,kg_map)

        mlp_in=torch.cat([h.squeeze(),batch_kg,self.mlp_history(batch_kg)],dim=1)
        mlp_hidden=self.mlp(mlp_in)
        mlp_hidden=torch.tanh(mlp_hidden)

        out=self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs,dim=1)

    def get_kg(self,batch_sequence,train_len,kg_map):
        """
        获取用户购买商品的嵌入的平均池化以获取当前的喜好表示
        :param batch_sequence: 用户的一系列购买记录
        :param train_len: 每个序列包含的batch数量
        :param kg_map: 商品的嵌入向量表
        :return: 平均池化之后的batch sequence，会降一维吧
        """
        batch_kg=[]
        # 下面就是做一个平均池化，获取当前的喜好
        for i,seq in enumerate(batch_sequence):
            # sequence中是内部id也就是0，1，2，3，因此需要用kg_map将他映射成嵌入向量
            seq_kg=kg_map[seq]
            seq_kg_avg=torch.sum(seq_kg,dim=0)
            seq_kg_avg=torch.div(seq_kg_avg,train_len[i])
            batch_kg.append(seq_kg_avg)

        # 将池化之后的数据再次拼接起来
        batch_kg=torch.stack(batch_kg)
        return batch_kg

    def RL_train(self, batch_sequences, items_to_predict, pred_one_hot, train_len, target_len):
        probs=[]
        probs_origin=[]
        each_sample=[]
        Rewards=[]
        input=self.item_embeddings(batch_sequences)

        out_enc,h=self.enc(input,train_len)

        kg_map=self.BN(self.kg_map)
        batch_kg=self.get_kg(batch_sequences,train_len,kg_map)

        mlp_in=torch.cat([h.squeeze(),batch_kg,self.mlp_history(batch_kg)],dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        out_distribution=F.softmax(out_fc,dim=1)
        probs_origin.append(out_distribution)
        out_distribution*=0.8
        # 将真实的标签的信息添加到预测的概率上
        out_distribution=torch.add(out_distribution,pred_one_hot)
        probs.append(out_distribution)

        # 设置文中的 pi(a|s)
        """
        这里是不是有点问题，Categorical里面的第一个参数是概率 但是这里的out_distribution
        是加上了pred_one_hot 的，就是说它的总和一定大于1，就不是个概率了
        """
        m=torch.distributions.categorical.Categorical(out_distribution)
        # 文中的action
        sample1=m.sample()
        each_sample.append(sample1)

        Reward,dist_sort=self.generateReward(sample1,self.args.T-1,3,items_to_predict,
                                             pred_one_hot,h,batch_kg,kg_map,target_len)
        Rewards.append(Reward)

        probs=torch.stack(probs,dim=1)
        probs_origin=torch.stack(probs_origin,dim=1)

        return probs, probs_origin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1), dist_sort

    def generateReward(self, sample1, path_len, path_num, items_to_predict,
                       pred_ont_hot, h_origin, batch_kg, kg_map, target_len):

        history_kg=self.mlp_history(batch_kg)
        Reward=[]
        dist=[]
        dist_replay=[]

        for paths in range(path_num):
            h=h_origin
            indexes=[]
            indexes.append(sample1)

            # 对未来时刻的一次预测，是内部id形式的
            dec_inp_index=sample1
            # 将内部id做一个embedding。与知识无关
            dec_inp=self.item_embeddings(dec_inp_index)
            dec_inp=dec_inp.unsqueeze(1)
            # 获取真实的知识图的平均池化
            ground_kg=self.get_kg(items_to_predict[:,self.args.T-path_len-1:], target_len, kg_map)

            for i in range(path_len):
                # 用预测的第一时刻放入gru以预测之后时刻
                out_enc,h=self.enc(dec_inp,h,one=True)

                mlp_in=torch.cat([h.squeeze(),batch_kg,self.mlp_history(batch_kg)],dim=1)
                mlp_hidden=self.mlp(mlp_in)
                mlp_hidden=torch.tanh(mlp_hidden)
                out_fc=self.fc(mlp_hidden)

                out_distribution=F.softmax(out_fc,dim=1)
                out_distribution*=0.8
                out_distribution=torch.add(out_distribution,pred_ont_hot)

                m=torch.distributions.categorical.Categorical(out_distribution)
                sample2=m.sample()

                dec_inp=self.item_embeddings(sample2)
                dec_inp=dec_inp.unsqueeze(1)
                indexes.append(sample2)
            indexes=torch.stack(indexes,dim=1)
            # 上面的获取到的是预测的用户会选择的商品
            episode_kg=self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)

            # 获取我们预测的商品的平均池化，与ground_truth求一个距离
            dist.append(self.cos(episode_kg,ground_kg))
            dist_replay.append(self.cos(episode_kg,history_kg))

            Reward.append(dcg_k(items_to_predict[:,self.args.T-path_len-1:],indexes,path_len+1))

        Reward=torch.FloatTensor(Reward).to(self.device)
        dist=torch.stack(dist,dim=0)
        dist=torch.mean(dist,dim=0)

        dist_replay=torch.stack(dist_replay,dim=0)
        dist_sort=self.compare_kgReward(Reward,dist_replay)

        Reward=torch.mean(Reward)
        Reward+=Reward+self.lamda*dist
        dist_sort=dist_sort.detach()
        return Reward,dist_sort

    def conpare_kgReward(self,reward,dist):
        """
        将dist按照reward的每列最大值的索引进行重新排个序

        :param reward: 和真实标签相比的奖励值
        :param dist: 和历史信息相比的奖励值
        :return: 排好序的dist
        """
        logit_reward,indice=reward.sort(dim=0)
        dist_sort=dist.gather(dim=0,index=indice)
        return dist_sort
