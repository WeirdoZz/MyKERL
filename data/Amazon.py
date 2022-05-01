from Dataset import Dataset

class Beauty(Dataset):
    def __init__(self):
        self.dir_path="./dataset/Amazon/Beauty/"
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
        :return: 训练集，测试集，用户总数，商品总数+index_shift，知识图映射
        """
        user_records=self.load_pickle(self.dir_path+self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        kg_mapping = self.load_kg(self.dir_path + self.kg_file)

        # 确保用户数量和商品数量是能够对上的
        assert self.num_users==len(user_mapping) and self.num_items==len(item_mapping)

        """
        下面这一句将用户购买记录中的商品的id全部都加上1，那不就不是这个用户
        买的商品了吗？
        """
        user_records=self.data_index_shift(user_records,increase_by=index_shift)

        train_set,test_set=self.split_data_sequentialy(user_records)
        return train_set,test_set,self.num_users,self.num_items+index_shift,kg_mapping
