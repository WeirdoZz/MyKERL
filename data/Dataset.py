import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import math


class Dataset(object):

    def load_pickle(self,name):
        """
        加载相关数据集的原始文件
        :param name: 数据集的所在位置
        :return: 返回加载后的文件
        """
        with open(name,"rb") as f:
            return pickle.load(f,encoding="latin1")

    def load_kg(self,name):
        """
        加载知识图
        :param name: 知识图所在文件的位置
        :return: 返回一个形状为[[...],...]的矩阵，每一行表示对应索引的商品id的知识表示
        """
        lines=open(name,"r").readlines()
        embedding_map=[]

        # 这里要去掉第一行，因为第一行是告诉你这个文件是几行几列的
        for index,l in enumerate(lines[1:]):
            # 移除开头和结尾的空格
            tmps=l.strip()
            # 去掉数据集中的第一列，那是序号列
            features=tmps.split(" ")
            features=features[1:]

            # 确保读出来的是50维的特征
            if len(features)==50:
                # 将特征添加到特征嵌入表中
                features=np.array([[float(i) for i in features]])
                embedding_map.append(features)
            # 将特征表concat成一个整体矩阵
            feature_matrix=np.concatenate(embedding_map,axis=0)

        """
        原代码在第一行加了一个零向量，但是我觉得没必要，他可能是为了配合前面做的index_shift，这个函数将商品序列的id都加了1
        对应到这个特征表中那每个特征向量的索引也要加1所以才加了一列0向量
        
        破案了,这里还是需要的,因为前面说了填充的0和本身的0会冲突的,所以需要将每个id+1,这样0号商品变成1号,而填充的0是不应该有知识表示的
        """
        zero_feature = np.zeros((1, 50))
        # 在特征矩阵的第一行增加一条零特征向量,因为0号商品实际上是填充的0,没有对应的知识
        feature_matrix = np.concatenate((zero_feature, feature_matrix), axis=0)
        return feature_matrix

    def save_pickle(self,obj,name,protocol=3):
        """
        将对象保存为pkl文件
        :param obj: 要保存的对象
        :param name: 保存文件的名称
        :param protocol: 保存使用的协议
        :return: 没有返回值
        """
        with open(name+'.pkl','wb') as f:
            pickle.dump(obj,f,protocol=protocol)

    def generate_inverse_mapping(self,data_list):
        """
        生成一个真实id到内部id的映射
        :param data_list: 保存了(内部id，真实id)的列表
        :return: 返回一个真实id->内部id的字典
        """
        inverse_mapping=dict()
        for inner_id,true_id in enumerate(data_list):
            inverse_mapping[true_id]=inner_id
        return inverse_mapping

    def convert_to_inner_index(self,user_records,user_mapping,item_mapping):
        """
        将用户的购买记录转成内部id，并且将每个用户id的排序从内部id为0开始依次往后
        :param user_records: 用户购买商品记录（原始id版）
        :param user_mapping: 用户内部id->真实id的映射
        :param item_mapping: 商品内部id->真实id的映射
        :return: 返回内部id为0开始的用户的购买记录(内部id版)和用户、商品的 真实id->内部id的映射表
        """
        inner_user_records=[]
        user_inverse_mapping=self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping=self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            # 将用户的内部id对应的真实id取出
            real_user_id=user_mapping[user_id]
            # 根据真实id取得该用户的购买商品序列
            item_list=list(user_records[real_user_id])
            # 将购买商品序列从真实id转换成内部id
            for index,real_item_id in enumerate(item_list):
                item_list[index]=item_inverse_mapping[real_item_id]
            # 将该用户的购买记录（内部id版）保存到列表中
            inner_user_records.append(item_list)

        return inner_user_records,user_inverse_mapping,item_inverse_mapping

    def split_data_randomly(self,user_records,seed=1999):
        """
        将每一个用户的购买记录随机划分为训练集和测试集
        :param user_records: 用户的购买记录(内部id版)
        :param seed: 随机种子
        :return: 返回分好的训练集和测试集
        """

        test_ratio=0.2
        train_set=[]
        test_set=[]
        for user_id,item_list in enumerate(user_records):
            # 将一个用户的购买记录中的一部分用于训练一部分用于测试
            tmp_train_sample,tmp_test_sample=train_test_split(item_list,test_size=test_ratio,random_state=seed)

            """
            下面的代码有问题，既然是从商品中随机选择训练和测试，那就破坏了
            本身购买顺序的时序性，会与文中提出的方法相悖
            
            破案了，这个方法是随机的方法，按顺序的方法在下面一个函数中
            """
            # 由于存在时序关系，所以商品的前后序列最好不要做变动
            train_sample=[]
            for place in item_list:
                if place not in tmp_test_sample:
                    train_sample.append(place)

            test_sample=[]
            for place in tmp_test_sample:
                test_sample.append(place)

            train_set.append(train_sample)
            test_set.append(test_sample)
        return train_set,test_set

    def split_data_sequentialy(self,user_records,test_ratio=0.2,seed=1999):
        """
        将每一个用户的购买记录随机划分为训练集和测试集
        :param user_records: 用户的购买记录(内部id版)
        :param test_ratio: 测试集所占的比例
        :param seed: 随机种子
        :return: 返回分好的训练集和测试集,还是按照原来的用户数量，只不过在序列上将其做了一个分割
        """
        train_set=[]
        test_set=[]

        for item_list in user_records:
            len_list=len(item_list)
            # 设置测试集所需要的比例
            num_test_sample=int(math.ceil(len_list*test_ratio))
            train_sample=[]
            test_sample=[]

            # 如果该用户的清单比较短，直接跳过
            if len_list<3:
                continue
            # 将最后几个购买的商品作为测试集
            for i in range(len_list-num_test_sample,len_list):
                test_sample.append(item_list[i])

            """
            这里创建训练集作者给出的代码是
            
            for place in item_list:
                if place not in set(test_sample):
                    train_sample.append(place)
                    
            这样的话如果前面买的东西后面又买了就会被去掉，还是会破坏顺序结构啊
            """
            for i in range(len_list-num_test_sample):
                train_sample.append(item_list[i])

            train_set.append(train_sample)
            test_set.append(test_sample)

        return train_set,test_set

    def generate_rating_matrix(self,train_set,num_users,num_items):
        """
        创建rating矩阵（用户数量，商品数量），其中行表示用户id，列表示商品id，买了为1，没买为0
        :param train_set: 训练集
        :param num_users: 用户总数
        :param num_items: 商品总数
        :return: 返回一个稀疏矩阵
        """
        row=[]
        col=[]
        data=[]

        for user_id,article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row=np.array(row)
        col=np.array(col)
        data=np.array(data)
        rating_matrix=csr_matrix((data,(row,col)),shape=(num_users,num_items))

        return rating_matrix

    def load_item_content(self,f_in,D=8000):
        """
        暂时不知道是干什么用的，后面用到它再说吧
        :param f_in:
        :param D:
        :return:
        """
        with open(f_in) as fp:
            lines=fp.readlines()
            X=np.zeros((len(lines),D))

            for i,line in enumerate(lines):
                strs=line.strip().split(" ")[2:]
                for str in strs:
                    segs=str.split(":")
                    X[i,int(segs[0])]=float(segs[1])

        return csr_matrix(X)

    def data_index_shift(self,lists,increase_by=2):
        """
        将lists中的每一个list中的每一个商品id都加上increase_by
        :param lists: 所有用户购买商品的信息列表
        :param increase_by: 将商品id增加多少
        :return: 返回增加id后的信息列表
        """
        for seq in lists:
            for i,item_id in enumerate(seq):
                seq[i]=item_id+increase_by

        return lists