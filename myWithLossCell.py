import mindspore
from mindspore import Tensor
import mindspore.nn as nn


class WithLossCell(nn.Cell):
    def __init__(self,model,criterion):
        super(WithLossCell,self).__init__(auto_prefix=False)
        self.model=model
        self.criterion=criterion
        self.output=None

    def get_output(self):
        return self.output

    def construct(self,im,label):
        self.output=self.model(im)
        loss=self.criterion(self.output,label)
        
        return loss
        
