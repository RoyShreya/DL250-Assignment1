import torch
import torchvision
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
g=open('Software1.txt','w')
print(sys.argv)
f=open(sys.argv[2],'r') 
a= f.read().splitlines()
print(a)
for i in range(0,len(a)):    
    if a[i].isdigit():
        num=int(a[i])
   
        if num%3==0 and num%5==0:
            g.write('fizzbuzz'+'\n')
        
        if num%3==0 and num%5!=0:
            g.write('fizz\n')
       
        if num%5==0 and num%3!=0:
            g.write('buzz\n')
        
        if num%5!=0 and num%3!=0:
            g.write(str(num)+'\n')
    
f.close()
g.close()


g=open('Software2.txt','w')

PATH="MyModel.pt"  
class MyNet(torch.nn.Module):
    def __init__(self, D_in, H1, D_out): #D_out is the number of classes
       
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H1, bias=True)
       
        self.linear2 = nn.Linear(H1, D_out, bias=True)
        self.bn = nn.BatchNorm1d(H1)

    def forward(self, x):
       
        h1_relu = F.relu(self.linear1(x))#.clamp(min=0)
        
        y_pred=self.linear2(self.bn(h1_relu))
        return y_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_X = np.genfromtxt('test_input.txt')
Xlist=[]
for sz in range(0,test_X.shape[0]):
     str1=(bin(int(test_X[sz])).replace("0b",""))
     a=np.zeros(10)
     for l in range(0,len(str1)):
         a[9-l]=int(str1[len(str1)-1-l])
     Xlist.append(a)
     #print(a)
inp=np.array(Xlist) 
print(inp.shape)
inp=np.reshape(inp,(inp.shape[0],inp.shape[1]))#,10)
#print(inp)

net=MyNet(10,100,4)  #inp dimension and hidden dimension and output dsimension ie. numof classes here
net=net.float()
    #net=net.cuda()
net.load_state_dict(torch.load(PATH)['model_state_dict'])  
inp=torch.as_tensor(inp, dtype=torch.float)
    
out=net(inp)#.cuda())
    #print(inp.size())
    #print(out.size())
    
label=np.argmax(np.array(out.detach().cpu().numpy().reshape(out.size(0),out.size(1))),axis=1)
inpArr=inp.detach().cpu().numpy()
    #print(label)
for i in range(0,label.shape[0]):
    if label[i]==1:
        g.write('fizz'+'\n')
    if label[i]==2:
        g.write('buzz'+'\n')
    if label[i]==0:
        g.write(str(int(test_X[i]))+'\n')
    if label[i]==3:
        g.write('fizzbuzz'+'\n')

f.close()            
g.close()
m=open('Software2.txt','r')
n=open('Software1.txt','r')
c=m.read().splitlines()
b=n.read().splitlines()
acc=0

for i in range(0,len(b)):
   if c[i]==b[i]:
       acc=acc+1
   
print("Accuracy in test set:")
print(acc/len(b))
f.close()
g.close()
