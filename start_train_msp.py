import sys
sys.path.append("..")
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops     
from mindspore.dataset import MnistDataset,Cifar10Dataset
import cv2
from mindspore.dataset.vision import Inter
import os,sys
from random import shuffle
from mindspore.train.callback import Callback,LossMonitor,TimeMonitor,ModelCheckpoint, CheckpointConfig
from mindspore import context,Model
from mindspore.ops import ExpandDims
import mindspore.numpy as np

#获得acc精确度
def get_acc(outputs,labels):
    batch_num=outputs.shape[0]
    predict_label=outputs.argmax(1)
    correct_num=(((predict_label==labels).astype("int")).sum()).asnumpy()
    acc=correct_num/batch_num
    return acc
    

#标签平滑
class LabelSmoothing(nn.Cell):
    def __init__(self,smoothing=0.1):
        super(LabelSmoothing,self).__init__()
        self.smoothing=smoothing

    def construct(self,inputs,targets):
        
        ls=ops.LogSoftmax(axis=1)
        log_probs=ls(inputs)
        
        num=inputs.shape[1]

        '''
        off_value=mindspore.Tensor(self.smoothing/(num-1.0), mindspore.float32)
        on_value=mindspore.Tensor(1-self.smoothing,mindspore.float32)
        onehot=ops.OneHot()
        y=onehot(targets,num,on_value,off_value)
        '''
        onehot=nn.OneHot(depth=num,axis=1,on_value=1-self.smoothing,off_value=self.smoothing/(num-1.0),dtype=mindspore.float32)
        y=onehot(targets)
        
        loss=(-y*log_probs).sum(1).mean()
        #loss=(-y*log_probs).mean(0).sum()
        return loss

class CNN(nn.Cell):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.SequentialCell(
            nn.Conv2d(1,16,kernel_size=3,pad_mode ="same",padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )

        self.layer2=nn.SequentialCell(
            nn.Conv2d(16,32,kernel_size=3,pad_mode ="same",padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.layer3=nn.SequentialCell(
            nn.Conv2d(32,64,kernel_size=3,pad_mode ="same",padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.layer4=nn.SequentialCell(
            nn.Conv2d(64,128,kernel_size=3,pad_mode ="same",padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.flatten = nn.Flatten()

        self.layer5=nn.SequentialCell(
            nn.Dense(128*12*12,1024),
            nn.ReLU(),
            nn.Dense(1024,128),
            nn.ReLU(),   #没有inplace
            nn.Dense(128,8)
            )
        

    def construct(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.flatten(x)
        #x=x.view(x.shape[0],-1)
        x=self.layer5(x)
        
        return x

'''
from utils import train
from utils import *
'''
import mindspore.dataset.vision.c_transforms as cv
from datadeal import *


def DataLoader(data,batch_size=16,repeat_siez=1,num_parallel_worker=1):
    dataset_generator = DatasetGenerator(data)
    dataset = mindspore.dataset.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
    
    #resize_op=cv.Resize((48,48), interpolation=Inter.LINEAR)  #插值方式
    #dataset = dataset.map(input_columns="data", operations=resize_op)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

train_data=DataLoader("E:/testdl/train")
test_data=DataLoader("E:/testdl/test")
'''
p=next(train_data.create_dict_iterator())
print(p["label"])
'''
mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")        #动态图PYNATIVE_MODE(类似pyhon一句一句执行)，静态图GRAPH_MODE(默认)

model=CNN()

criterion=LabelSmoothing()#nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer=mindspore.nn.Adam(params=model.trainable_params(), learning_rate=0.001)

'''
#############使用mindspore框架提供的API来训练数据
#保存模型
ckc=CheckpointConfig(save_checkpoint_steps=40, keep_checkpoint_max=6)#keep_checkpoint_max=6保存最大文件数为6
mc=ModelCheckpoint(prefix="cnn", directory="pth", config=ckc)

model=Model(model,criterion,optimizer,metrics={"Accuracy": nn.Accuracy()})

model.train(epoch=10,train_dataset=train_data,callbacks=[LossMonitor(23),mc], dataset_sink_mode=False)#
'''

'''
##############使用自己定义的循环来训练数据
net=nn.WithLossCell(model,criterion)
net=nn.TrainOneStepCell(net,optimizer)
net.set_train(True)
for epoch in range(10):
    epoch_loss=0.0
    for data_dict in train_data.create_dict_iterator():
        im=data_dict["data"].astype('float32')
        label=data_dict["label"]
        loss=net(im,label)
        epoch_loss+=loss
        
    print("Epoch: %d loss :%f"%(epoch+1,(epoch_loss/23).asnumpy()))
'''


from myWithLossCell import *
from myTrainOneStepCell import *

net=WithLossCell(model,criterion)
net=TrainOneStepCell(net,optimizer)
#net.set_train(True)
for epoch in range(30):
    epoch_loss=0.0
    epoch_acc=0.0
    for data_dict in train_data.create_dict_iterator():
        im=data_dict["data"].astype('float32')
        label=data_dict["label"]
        
        loss,output=net(im,label)
        
        epoch_loss+=loss
        acc=get_acc(output,label)
        epoch_acc+=acc
    print("Epoch: %d loss :%f acc: %f"%(epoch+1,(epoch_loss/23).asnumpy(),epoch_acc/23))
