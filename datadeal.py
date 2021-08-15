import mindspore
import os,cv2
import numpy as np
from PIL import Image
from mindspore import dtype
from random import shuffle

class DatasetGenerator:
    def __init__(self,filename,transform=None):
        self.images=[]
        self.labels=[]
        self.transform=transform
        l=-1
        self.dat=[]
        for root,dirs,files in os.walk(filename):
            for filename in (x for x in files if x.endswith(".jpg")):
                filepath=os.path.join(root,filename)
                


                image=cv2.imread(filepath,0)
                
                name=filepath.split("\\")[-1]
                self.labels.append(l)
               
                data=image
                #data=LBP(data)
                #data=huiduhua(data)
                data = cv2.resize(data, (48, 48))
                data=np.expand_dims(data,2)
                self.images.append(data)
            l+=1
        for i,j in  zip(self.images,self.labels):
            self.dat.append([i,j])
        self.sshuffle()

    def __getitem__(self,index):
        img = self.images[index].astype('float32')
        img = img.transpose((2,0,1))
        if self.transform is not None:
            img = self.transform(img)
        #label = mindspore.Tensor([self.labels[index]],dtype.float32)
        label = self.labels[index]
        return img/255, label

    def __len__(self):
        return len(self.images)

    def __len__(self):
        return len(self.images)
    #对数据集进行打乱
    def sshuffle(self):
        shuffle(self.dat)
        self.dat=np.array(self.dat)
        self.images.clear()
        self.labels.clear()
        for i,j in zip(self.dat[:,0],self.dat[:,1]):
            self.images.append(i)
            self.labels.append(j)
