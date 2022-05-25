import time

import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))

    def load(self,path,change_opt=True):
        """
        加载之前训练好的模型
        :param path: 模型所在的路径
        :param change_opt: 是否改变opt中的模型
        :return: 返回之前加载好的模型
        """
        print(path)
        data=torch.load(path)

        if "opt" in data:
            if change_opt:
                self.opt.parse(data['opt'],print_=False)

            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self,name=None,new=False):
        """
        将训练好的模型保存起来
        :param name: 保存的文件起的文件名
        :param new: 不知道什么意思，反正没影响
        :return: 返回模型保存的位置
        """
        prefix="checkpoints/"+self.model_name+"_"+self.opt.type_+"_"
        if name is None:
            name=time.strftime("%m%d_%H:%M:%S.pth")
        path=prefix+name

        if new:
            data={"opt":self.opt.state_dict(),"d":self.state_dict()}
        else:
            data=self.state_dict()

        torch.save(data,path)
        return path

    def get_optimizer(self,lr1,lr2=0,weight_decay=0):
        """
        就是设置优化器的参数
        :param lr1:
        :param lr2:
        :param weight_decay:
        :return: 返回设置好的优化器
        """
        # 生成一个迭代器，将所有的模型参数编号
        ignored_params=list(map(id,self.embed.parameters()))
        # 生成一个不在ignored_params中的参数的迭代器
        base_params=filter(lambda p:id(p) not in ignored_params,self.parameters())

        if lr2 is None:
            lr2=lr1*0.5
        optimizer=torch.optim.Adam([
            dict(params=base_params,weight_decay=weight_decay,lr=lr1),
            {"params":self.embed.parameters(),"lr":lr2}
        ])

        return optimizer