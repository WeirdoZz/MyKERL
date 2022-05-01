import torch.nn as nn


class KERL(nn.Module):
    def __init__(self,num_user,num_items,model_args,device,kg_map):
        super(KERL, self).__init__()

