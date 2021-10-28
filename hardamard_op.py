import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()#实现父类的初始化
        self.conv1=nn.Conv2d(3,6,3)#定义卷积层组件
        self.pool1=nn.MaxPool2d(2,2)#定义池化层组件
        self.conv2=nn.Conv2d(6,16,3)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*5*5,120)#定义线性连接
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.had=Hadmar(8)

    def forward(self,x):#x模型的输入
        x=self.had(x)
        x=self.conv1(x)
        x=self.pool1(F.relu(x))
        x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)#表示将x进行reshape，为后面做为全连接层的输入
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class Hadmar(nn.Module):
    def __init__(self,order:int=8):
        super(Hadmar, self).__init__()
        self.order=order
        self.hadmar=np.ones((order, order), dtype = np.longlong)
        for i in range(self.order):
            for j in range(self.order):
                self.hadmar[i][j] = self.Creat_Hadmard(i,j)
        self.hadmar=torch.from_numpy(self.hadmar)
        self.hadmar=torch.stack((self.hadmar,self.hadmar,self.hadmar),2)

    def Creat_Hadmard(self,i = 4, j = 4):
        temp = i & j
        result = 0
        for step in range(4):
            result += ((temp >> step) & 1)
        if 0 == result % 2:
            sign = 1
        else:
            sign = -1
        return sign

    def forward(self,input):
        output=torch.mm(input,self.hadmar)
        return output

    def extra_repr(self):
        return 'order={}, hardma={}'.format(
            self.order, self.hardma is not None
        )

data = [[1, 2,3,4],[5,6,7,8],[3, 4,5,7],[8,9,10,11]]
x_data = torch.tensor(data)
net=Net()
input = torch.randn(8, 8, 3)
output=net(input)
print(output)