import argparse
from data import Amazon

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    """
    L:最大的序列长度
    T:事件的长度
    """

    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=3)

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

    maxlen=0
    # 由于用户购买力不同，训练集的每组数据的条数不定相同
    for inter in train_set:
        if len(inter)>maxlen:
            maxlen=len(inter)

    train=Interactions(train_set,num_users,num_items)
