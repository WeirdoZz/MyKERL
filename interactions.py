import numpy as np
import scipy.sparse as sp


class Interactions(object):
    def __init__(self, user_item_sequence, num_users, num_items):
        user_ids, item_ids = [], []
        for uid, item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        # 是一个[1,1,1,1,...,5,5,5,5,]这样的列表，其中每个值就是对应的id，出现的次数就是序列的长度
        user_ids = np.asarray(user_ids)
        # 是一个[2,7,4,,8,...,12,334,..]的列表，其中每个值是商品id，和上面这个user_id列表是长度相同的
        item_ids = np.asarray(item_ids)

        # 用户总数量和商品总数量
        self.num_users = num_users
        self.num_items = num_items

        self.user_ids = user_ids
        self.item_ids = item_ids

        self.sequences = None
        self.valid_sequences = None

    def __len__(self):
        return len(self.user_ids)

    def tocoo(self):
        """
        转换成稀疏矩阵
        :return: 用户商品之间的关系矩阵
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)), shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        将coo矩阵转换成csr矩阵保存形式
        :return: csr矩阵保存形式
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        转换成序列的形式，根据两个参数确定合法的子序列以及其标签
        :param sequence_length: 序列长度
        :param target_length: 标签长度
        :return:
        """

        # 先根据用户id进行一个排序,按照升序返回对应的索引
        # 相当于对用户的顺序做了一个限制而已
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        # 按照上面的顺序获取所有用户id(不重复)，并且获取其第一次出现的索引，和每个用户出现的次数（也就是购买商品的序列长度）
        user_ids, indices, counts = np.unique(user_ids, return_index=True, return_counts=True)

        # 一共能生成多少个长度为sequence_length的子序列，如果序列长度大于sequence_length那没话说，但是如果序列长度小于sequence_length
        # 就只能全取了
        num_subsequences = sum([c - sequence_length + 1 if c >= sequence_length else 1 for c in counts])
        # 做成一个 数量×长度的数组 （对应的标签也是如此）
        sequences = np.zeros((num_subsequences, sequence_length), dtype=np.int64)
        # 保存每个子序列对应的真实预测序列
        sequences_targets = np.zeros((num_subsequences, target_length), dtype=np.int64)
        # 保存每个子序列属于那个user
        sequences_users = np.empty(num_subsequences, dtype=np.int64)
        # 保存每个子序列的长度
        sequences_length = np.empty(num_subsequences, dtype=np.int64)
        # 保存每个子序列的真实预测序列的长度
        sequences_traget_length = np.empty(num_subsequences, dtype=np.int64)

        valid_sequences = np.zeros((self.num_users, sequence_length), dtype=np.int64)
        valid_users = np.empty(self.num_users, dtype=np.int64)
        valid_target_sequences=np.zeros((self.num_users,target_length),dtype=np.int64)
        valid_length=np.empty(self.num_users,dtype=np.int64)
        valid_target_length=np.empty(self.num_users,dtype=np.int64)

        _uid = None
        for i, (uid, item_seq) in enumerate(_generate_sequences(user_ids,
                                                                item_ids,
                                                                indices,
                                                                sequence_length)):
            if uid != _uid:
                valid_sequences[uid][:] = item_seq[:sequence_length-target_length]
                valid_target_sequences[uid][:]=item_seq[-target_length:]
                valid_users[uid] = uid
                valid_length[uid]=sequence_length-target_length
                valid_target_length[uid]=target_length
                _uid = uid
                continue

            sequences_targets[i][:] = item_seq[-target_length:]
            sequences_traget_length[i] = target_length
            sequences[i][:] = item_seq[:sequence_length-target_length]
            sequences_length[i] = sequence_length-target_length
            sequences_users[i] = uid

        self.sequences = SequenceInteractions(sequences_users, sequences, sequences_targets, sequences_length,
                                              sequences_traget_length)
        self.valid_sequences = SequenceInteractions(valid_users, valid_sequences, valid_target_sequences, valid_length,
                                                    valid_target_length)

    def to_newsequence(self, sequence_length=5, target_length=1):

        max_sequence_length = sequence_length + target_length
        # 先根据用户id进行一个排序,按照升序返回对应的索引
        # 相当于对用户的顺序做了一个限制而已
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        # 按照上面的顺序获取所有用户id，并且获取其第一次出现的索引，和每个用户出现的次数（也就是购买商品的序列长度）
        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        seq_user = []
        sequences = []
        sequences_targets = []
        sequences_length = []
        sequences_targetlen = []
        valid_users = []
        valid_sequences = []
        valid_seq_length = []

        for i, user_id in enumerate(user_ids):
            start_idx = indices[i]
            try:
                stop_idx = indices[i + 1]
            except:
                stop_idx = None

            # 用户的整个购买商品序列
            one_sequence = item_ids[start_idx:stop_idx]

            """
            这里原代码给的情况理论上是不可能发生的
            
            # 如果购买序列不存在，这一步应该是不可能发生的
            if len(one_sequence) <= 0:
                print(user_id, one_sequence, indices[i])
            """
            """
            这一步其实也是不需要的
            
            # 如果购买序列比给定序列长度长，获取其最后一个序列
            if len(one_sequence) > sequence_length:
                one_sequence_valid = one_sequence[-sequence_length:]    
            """
            # 如果长度不够sequence_length就需要填补，如果够了这一步填补相当于不做
            valid_train_seq = np.pad(one_sequence[-sequence_length:], (0, sequence_length - len(one_sequence[-sequence_length:])), 'constant')
            valid_users.append(user_id)
            valid_sequences.append(valid_train_seq)
            valid_seq_length.append(len(one_sequence[-sequence_length:]))

            for train_len in range(len(valid_train_seq) - 1):
                # 获取该序列中的子序列
                sub_seq = valid_train_seq[0:train_len + 1]
                # 保存子序列的长度
                sequences_length.append(train_len + 1)
                # 需要填充的数量
                num_paddings = sequence_length - train_len - 1
                sub_seq = np.pad(sub_seq, (0, num_paddings), 'constant')
                # 标签子序列
                target_sub = valid_train_seq[train_len + 1:train_len + 1 + target_length]
                target_len = len(target_sub)
                target_sub = np.pad(target_sub, (0, target_length - len(target_sub)), 'constant')

                seq_user.append(user_id)
                sequences.append(sub_seq)
                sequences_targetlen.append(target_len)
                sequences_targets.append(target_sub)

        sequence_users = np.array(seq_user)
        sequences = np.array(sequences)
        sequences_length = np.array(sequences_length)
        sequences_targetlen = np.array(sequences_targetlen)
        sequences_targets = np.array(sequences_targets)

        valid_users = np.array(valid_users)
        valid_sequences = np.array(valid_sequences)
        valid_seq_length = np.array(valid_seq_length)

        self.sequences = SequenceInteractions(sequence_users, sequences, targets=sequences_targets,
                                              length=sequences_length, tar_len=sequences_targetlen)
        self.valid_sequences = SequenceInteractions(valid_users, valid_sequences, length=valid_seq_length)


class SequenceInteractions(object):
    def __init__(self, user_ids, sequences, targets=None, length=None, tar_len=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets
        self.length = length
        self.tarlen = tar_len

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _generate_sequences(user_ids, item_ids, indices, sequence_length):
    # 遍历每个user_id第一次出现的索引
    for i in range(len(indices)):
        # 该索引作为开始索引
        start_idx = indices[i]

        if i == len(indices) - 1:
            stop_idx = None
        else:
            # 结束索引是其下一个索引
            # 因为user_id是已经排过序的，所以其第一次出现的索引两个索引
            # 之间就是真实的user_id所拿商品的的全部序列
            stop_idx = indices[i + 1]

        # 用时间窗进行滑动，获取子序列
        for sub_seq in _sliding_window(item_ids[start_idx:stop_idx], sequence_length):
            yield user_ids[i], sub_seq


def _sliding_window(tensor, window_size):
    # 如果序列够长，就用时间窗在上面滑动
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor) - window_size + 1):
            yield tensor[i:i + window_size]
    else:
        # 否则序列不够长就填充后作为一条序列返回就可以了
        num_paddings = window_size - len(tensor)
        yield np.pad(tensor, (0, num_paddings), "constant")
