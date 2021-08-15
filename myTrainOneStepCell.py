import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple, Parameter
import mindspore.ops.functional as F
from mindspore.ops import ExpandDims


class TrainOneStepCell(nn.Cell):
    def __init__(self,net,optimizer,sens=1.0):
        super(TrainOneStepCell,self).__init__(auto_prefix=False)

        self.net=net
        # 使用tuple包装weight
        self.weights=ParameterTuple(net.trainable_params())
        
        # 使用优化器
        self.optimizer=optimizer
        # 定义梯度函数
        self.grad=ops.GradOperation(get_by_list=True,sens_param=True)
        self.sens=sens

    def construct(self,im,label):

        weights=self.weights
        
        loss=self.net(im,label)
        
        
        # 为反向传播设定系数

        #生成一个与loss的shape一样的张量，并且内容是1.0
        sens=ops.Fill()(ops.DType()(loss),ops.Shape()(loss),self.sens)
        
        grad=self.grad(self.net,weights)(im,label,sens)

        loss=F.depend(loss,self.optimizer(grad))

        return loss,self.net.get_output()

