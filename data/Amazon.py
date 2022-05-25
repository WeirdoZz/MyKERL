from data.Dataset import Dataset

class Beauty(Dataset):
    def __init__(self):
        self.dir_path="data/dataset/Amazon/Beauty/"
        self.user_record_file = 'Beauty_item_sequences.pkl'
        self.user_mapping_file = 'Beauty_user_mapping.pkl'
        self.item_mapping_file = 'Beauty_item_mapping.pkl'
        self.kg_file = 'embedding.txt'

        self.num_users=22363
        self.num_items=12101
        self.vocab_size=0

        self.user_records=None
        self.user_mapping=None
        self.item_mapping=None

    def generate_dataset(self,index_shift=1):
        """
        生成需要的数据集

        :param index_shift: 控制商品id增加多少的
        :return: 训练集（[[...],...]），测试集([[...],...])，用户总数，商品总数，知识图映射([[...],...])
        """
        # 用户的购买序列，是[[...],[...]]这样的一个列表，内容是商品的id
        user_records=self.load_pickle(self.dir_path+self.user_record_file)
        # 就是一个用户的序列号的列表(比如 A165SAZZONY1P4这种)，共22363个用户
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        # 就是一个商品序列号的列表(比如 B726V128这种)，共12101个商品
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        kg_mapping = self.load_kg(self.dir_path + self.kg_file)

        # 确保用户数量和商品数量是能够对上的
        assert self.num_users==len(user_mapping) and self.num_items==len(item_mapping)

        """
        下面这一句将用户购买记录中的商品的id全部都加上1，那不就不是这个用户
        买的商品了吗？
        
        这里我想明白了,因为后面序列填充的时候用的是0,如果说这里不对每个商品加1的话,那真实商品的embedding就会和填充的0相冲突了啊
        """
        user_records = self.data_index_shift(user_records, increase_by=index_shift)
        # 将序列进行划分，获取训练集和测试集，这里两个集合的形状都是[[...],[...],...] 0维长度相同，1维不相同
        train_set,test_set=self.split_data_sequentialy(user_records)
        """
        这里因为index_shift,所以返回的时候要多加一个index_shift,否则就出现问题了
        """
        return train_set, test_set, self.num_users, self.num_items + index_shift, kg_mapping
