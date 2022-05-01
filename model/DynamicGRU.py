import torch
from torch.autograd import Variable

from BasicModule import *

class DynamicGRU(BasicModule):
    def __init__(self,input_dim,output_dim,num_layers=1,
                 bidirectional=False,batch_first=True):
        super(DynamicGRU, self).__init__()
        self.embed_dim=input_dim
        self.hidden_dim=output_dim
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.batch_first=batch_first
        self.gru=nn.GRU(self.embed_dim,
                        self.hidden_dim,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional,
                        batch_first=self.batch_first)
    def forward(self,inputs,lengths,one=False):
        """
        :param inputs: 形状为(b,l,e)，是当前时刻的经过pad填充过的输入
        :param lengths: 长度为b,是每个batch的未填充时的序列数量
        :param one: 是否只有一条数据吗？
        :return: 返回输出和隐状态
        """
        if one==True:
            hidden=lengths
            out,ht=self.gru(inputs,hidden)
        else:
            # 将每个batch的sample数量进行排序，idx_sort就是从高到低的索引值
            _,idx_sort=torch.sort(lengths,dim=0,descending=True)
            # 对排好序的索引再进行依次进行排序，idx_unsort的索引就是原本所在的位置
            # 对应的值就是现在的位置
            """
            这里可以自己画一个示例就非常好理解了
            """
            _,idx_unsort=torch.sort(idx_sort,dim=0)

            # 从输入的每个batch中选择，将数据多的batch放到前面，不然后面压缩序列的时候会报错
            sort_embed_input=inputs.index_select(0,Variable(idx_sort))
            # 将lengths本身也按照从高到低的顺序排序，才能和上面的排过序的batch对应上
            sort_lengths=lengths[idx_sort]

            # 对输入的序列进行压紧，避免那些无效的计算过程
            inputs_packed=nn.utils.rnn.pack_padded_sequence(sort_embed_input,
                                                            sort_lengths,
                                                            batch_first=True)
            out_pack,ht=self.gru(inputs_packed)

            # 对压紧的输出进行填充回来
            out=nn.utils.rnn.pad_packed_sequence(out_pack,batch_first=True)
            out=out[0]

            # 将结果的顺序转变成原来的顺序
            ht=torch.transpose(ht,0,1)[idx_unsort]
            ht=torch.transpose(ht,0,1)

            out=out[idx_unsort]
        return out,ht