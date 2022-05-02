import numpy as np
import scipy.sparse as sp


class Interactions(object):
    def __init__(self,user_item_sequence,num_users,num_items):
        user_ids,item_ids=[],[]
        for uid,item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        user_ids=np.asarray(user_ids)
        item_ids=np.asarray(item_ids)

        self.num_users=num_users
        self.num_items=num_items

        self.user_ids=user_ids
        self.item_ids=item_ids

        self.sequences=None
        self.test_sequences=None

    def __len__(self):
        return len(self.user_ids)

    def tocoo(self):
        """
        转换成稀疏矩阵
        :return: 用户商品之间的关系矩阵
        """

        row=self.user_ids
        col=self.item_ids
        data=np.ones(len(self))

        return sp.coo_matrix((data,(row,col)),shape=(self.num_users,self.num_items))

    def tocsr(self):
        """
        将coo矩阵转换成csr矩阵保存形式
        :return: csr矩阵保存形式
        """

        return self.tocoo().tocsr()

    def to_sequence(self,sequence_length=5,target_length=1):
        """
        转换成序列的形式，根据两个参数确定合法的子序列以及其标签
        :param sequence_length: 序列长度
        :param target_length: 标签长度
        :return:
        """

        max_sequence_length=sequence_length+target_length

        # 先根据用户id进行一个排序,按照升序返回对应的索引
        # 相当于对用户的顺序做了一个限制而已
        sort_indices=np.lexsort((self.user_ids,))

        user_ids=self.user_ids[sort_indices]
        item_ids=self.item_ids[sort_indices]

        # 按照上面的顺序获取所有用户id，并且获取其第一次出现的索引，和每个用户出现的次数（也就是购买商品的序列长度）
        user_ids,indices,counts=np.unique(user_ids,return_index=True,return_counts=True)

        # 一共能生成多少个长度为sequence_length的子序列
        num_subsequences=sum([c-max_sequence_length+1 if c>=max_sequence_length else 1 for c in counts])
        # 做成一个 数量×长度的数组 （对应的标签也是如此）
        sequences=np.zeros((num_subsequences,sequence_length),dtype=np.int64)
        sequences_targets=np.zeros((num_subsequences,target_length),dtype=np.int64)
        sequences_users=np.empty(num_subsequences,dtype=np.int64)

        test_sequences=np.zeros((self.num_users,sequence_length),dtype=np.int64)
        test_users=np.empty(self.num_users,dtype=np.int64)

        _uid=None
        for i,(uid,item_seq) in enumerate(_generate_sequences())


def _generate_sequences(user_ids,item_ids,indices,max_sequence_length):

    for i in range(len(indices)):
        start_idx=indices[i]

        if i>=len(indices)-1:
            stop_idx=None
        else:
            stop_idx=indices[i+1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],max_sequence_length):
            yield (user_ids[i],seq)

def _sliding_window(tensor,window_size):
    if len(tensor)-window_size>=0:
        i=len(tensor)
        yield tensor[i-window_size:i]
    else:
        num_paddings=window_size-len(tensor)
        yield np.pad(tensor,(0,num_paddings),"constant")