import torch
import torchvision
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
epoch=100#0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X = np.genfromtxt('train_input.txt')
Xlist=[]
for sz in range(0,train_X.shape[0]):
     str1=(bin(int(train_X[sz])).replace("0b",""))
     a=np.zeros(10)
     for l in range(0,len(str1)):
         a[9-l]=int(str1[len(str1)-1-l])
     Xlist.append(a)
     print(a)
train_X=np.array(Xlist) 
print(train_X)    
train_label=[]
f=open('train_label.txt')
a= f.read().splitlines()
c0=1.0
c1=1.0
c2=1.0
c3=1.0
for i in range(0,len(a)):    
    if a[i]=='num':
        train_label.append(0)
        c0=c0+1
    if a[i]=='fizz':
        train_label.append(1)
        c1=c1+1
    if a[i]=='buzz':
        train_label.append(2)
        c2=c2+1
    if a[i]=='fizzbuzz':   
        train_label.append(3)
        c3=c3+1
PATH="MyModel.pt"     
print(c0)
print(c1)
print(c2)
print(c3)   
f.close()    
train_label=np.array(train_label)
class MyNet(torch.nn.Module):
    def __init__(self, D_in, H1,H2, D_out): #D_out is the number of classes
       
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H1, bias=True)
        self.linear2 = nn.Linear(H1, H2, bias=True)
        self.linear3 = nn.Linear(H2, D_out, bias=True)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)

    def forward(self, x):
       
        h1_relu = F.relu(self.linear1(x))#.clamp(min=0)
        h2_relu = F.relu(self.linear2(self.bn1(h1_relu)))
        y_pred=self.linear3(self.bn2(h2_relu))
        return y_pred

class MyDataset(Dataset):

    """
    A customized data loader.
    """
    def __init__(self, root):
        """ Intialize the dataset
        """
        self.input = torch.as_tensor(train_X,dtype=torch.float).unsqueeze(1)
        self.label=  torch.as_tensor(train_label,dtype=torch.int).unsqueeze(1)
        self.root=root
        print(self.label.size())
        print(self.input.size())
        self.len = (self.input.size(0))
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if index<self.len :
            
            return self.input[index],self.label[index]
        else:
            return NULL

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
MyDataset(os.getcwd())


def train(BATCH_SIZE,D_in, H1,H2, D_out):
    full_dataset=MyDataset(os.getcwd())
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader.len=train_size
    val_loader.len=test_size
    train_loss_list=[]
    val_loss_list=[]
    acc_list=[]
    print(type(device))
    if torch.cuda.is_available():
        net=MyNet(D_in,H1,H2,D_out)  #inp dimension and hidden dimension and output dsimension ie. numof classes here
        net=net.float()
        net=net.cuda()
        for iteration in range(0,epoch):
                
                if(iteration==0):
                    if os.path.exists(PATH):
                        net.load_state_dict(torch.load(PATH)['model_state_dict'])  
                
                learning_rate = .005#1#00001
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=1e-4)
                #W=np.array([1/c0,1/c1,1/c2,1/c3])
                W=np.array([1,1,1,1])
                W=torch.as_tensor(W,dtype=torch.float)
                W=W.cuda()
                criterion=nn.CrossEntropyLoss(weight=W)
                j=0
                for batch_index,(inp,label)  in enumerate(train_loader,0) : 
                    inp=torch.as_tensor(inp, dtype=torch.float).view(BATCH_SIZE,D_in)
                    label=torch.as_tensor(label, dtype=torch.long).view(BATCH_SIZE,1)
                    #print(inp)
                    inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                    label=label.to(device='cuda')#, dtype=torch.double)
                    out=net(inp)
                    out=out.unsqueeze(2)
                    
                    optimizer.zero_grad()
                    loss = criterion(out,label)
                    train_loss_list.append(loss.item())
                    np.save('TrainLoss.npy',np.array(train_loss_list))
                    loss.backward()
                    optimizer.step()
                val_loss=0.0
                acc=0.0
                
                with torch.no_grad():
                    j=0
                    
                    for batch_index,(inp,label)  in enumerate(val_loader,0):
                         inp=torch.as_tensor(inp, dtype=torch.float).view(BATCH_SIZE,D_in)
                         label=torch.as_tensor(label, dtype=torch.long).view(BATCH_SIZE,1)
                         inp=inp.to(device='cuda')# dtype=torch.double) #double model weight so make all input float tensor to double
                         label=label.to(device='cuda')#, dtype=torch.double)
                         out=net(inp)
                         out=out.unsqueeze(2)
                         batch_acc=0.0
                         for b in range(0,BATCH_SIZE):
                             if np.argmax(np.array(out[b,:,0].detach().cpu().numpy().reshape(1,out.size(1),out.size(2))),axis=1)==label[b,0].item():
                                 #  print("T")
                                 batch_acc=batch_acc+1
                             #else:
                                 #print("F")
                         acc=acc+batch_acc
                         val_loss=val_loss+criterion(out,label).item()
                         j=j+1
                    #print(j)
                    #print(test_size)
                    avg_val_loss=val_loss/j 
                  #  print(acc)
                    #print(test_size)
                    avg_acc=acc/test_size
                    val_loss_list.append(avg_val_loss)
                    np.save('ValLoss.npy',np.array(val_loss_list)) 
                   # print('iter no:'+str(epoch))
                   # print('Validation Loss:')
                    #print(avg_val_loss) 
                    #print('accuracy') 
                    acc_list.append(avg_acc)
                    print(avg_acc) 
                    np.save('ValAcc.npy',np.array(acc_list)) 
                torch.save({'epoch': iteration,'model_state_dict':net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, PATH)   

train(60,10,100,100,4)
