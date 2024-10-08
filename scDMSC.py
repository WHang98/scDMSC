"""
By Xifeng Guo (guoxifeng1990@163.com), May 13, 2020.
All rights reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import *
import utils
import scipy.io as sio
import math
from torch.nn.parameter import Parameter
from preprocess import read_dataset, normalize, clr_normalize_each_cell
import  scanpy   as sc
import h5py
from datetime import datetime
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocess import *


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-4 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient,x)
        return y

class MSCNet(nn.Module):
     def __init__(self,X1_dim, X2_dim ,X_dim,z_dim,N,M,n,n_clusters,sigma=2.5):
        super().__init__()
        self.self_expression = SelfExpression(n)
        self.sigma = sigma  
        self.fc1=nn.Linear(X_dim, N)
        self.fc2=nn.Linear(N, M)
        self.fc3 = nn.Linear(M, z_dim)
        self.fc4 = nn.Linear(z_dim, M)
        self.fc5 = nn.Linear(M, N)
        self.fc61 = nn.Linear(N, X1_dim) 
        self.fc62 = nn.Linear(N, X2_dim) 
        self.n_clusters = n_clusters
     def encode(self, x):
        x = x.to(torch.float32)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        z=self.fc3(h2)
        return z
        
 
     def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h51 = F.relu(self.fc5(h4))
        h52 = F.relu(self.fc5(h4))
        return self.fc61(h51), self.fc62(h52)                         

     def forward(self,x):
        z = self.encode(x+torch.randn_like(x) * self.sigma) 
        zc = self.self_expression(z)
        x1_rec,x2_rec=self.decode(z)
        return  z, zc, x1_rec,x2_rec
     def encodeBatch(self, X):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        inputs = Variable(X)
        z,zc,_,_=  self.forward(inputs)
        return z
     def loss_fn(self,x,x1,x2,x1_rec,x2_rec,z, zc, weight_ae,weight_trace, weight_selfExp,weight_C):

        loss1_ae = weight_ae*torch.sum(torch.square(torch.subtract(x1_rec, x1)))
        loss2_ae = weight_ae*torch.sum(torch.square(torch.subtract(x2_rec, x2)))
        loss_ae=loss1_ae +loss2_ae  
        x1_inputfla = torch.reshape(x1, [n, -1])
        x1_inputfla = x1_inputfla.to(torch.float32)
        x1_recfla = torch.reshape(x1_rec, [n, -1])
        x1_recfla = x1_recfla.to(torch.float32)
        x2_inputfla = torch.reshape(x2, [n, -1])
        x2_inputfla = x2_inputfla.to(torch.float32)
        x2_recfla = torch.reshape(x2_rec, [n, -1])
        x2_recfla = x2_recfla.to(torch.float32)
        normL = True
        absC = torch.abs(self.self_expression.Coefficient)
        C = (absC + absC.T) * 0.5
        C = C + torch.eye(self.self_expression.Coefficient.shape[0])

        if normL == True:
            D = torch.diag(1.0 / torch.sum(C,axis=1))
            I = torch.eye(D.shape[0])
            L = I - torch.matmul(D,C)
            D = I
        else:
            D = torch.diag(torch.sum(C, axis=1))
            L = D - C
        XLX_r1 = torch.matmul(torch.matmul((x1_inputfla.T),L),x1_recfla)
        XLX_r2 = torch.matmul(torch.matmul((x2_inputfla.T),L),x2_recfla)
        X1sub = x1_inputfla - x1_recfla
        X2sub = x2_inputfla - x2_recfla
        tracelossx1 =torch.sum(torch.square(X1sub)) +  2.0 * torch.trace(XLX_r1)#/self.batch_size
        tracelossx2 =torch.sum(torch.square(X2sub)) +  2.0 * torch.trace(XLX_r2)#/self.batch_size
        tracelossx =tracelossx1+ tracelossx2
        loss_selfExp = torch.sum(torch.square(torch.subtract(zc, z)))
        norm = torch.norm(self.self_expression.Coefficient,keepdim=True)
        penalty = torch.matmul(norm, norm)
        loss_s=torch.sum(torch.abs(self.self_expression.Coefficient))+ torch.trace(torch.abs(self.self_expression.Coefficient))
        loss_sc = weight_ae*loss_ae + weight_trace * tracelossx+ weight_selfExp * loss_selfExp#+penalty+loss_s#+weight_C*Contrast
        loss_sc /= x1.size(0)  # just control the range, does not affect the optimization.
        return loss_sc,loss_ae, tracelossx,loss_selfExp,penalty,loss_s

     def pretrian(self, x1,x2,x ,epoch = 100,batch_size=3762,lr=0.002):   
            self.train()
            time = datetime.now().strftime('%Y%m%d')
            dataset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(x))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
            for epoch in range(epoch):
                for batch_idx, (x1_batch,x2_batch,x_batch,) in enumerate(dataloader):

                    x_tensor = Variable(x_batch)
                    x1_tensor = Variable(x1_batch)
                    x2_tensor = Variable(x2_batch)
                    _, _, x1_rec, x2_rec = self.forward(x_tensor)
                    loss1 = torch.sum(torch.square(torch.subtract(x1_rec, x1_tensor)))/x1_tensor.size(0)
                    loss2 = torch.sum(torch.square(torch.subtract(x2_rec, x2_tensor)))/x2_tensor.size(0)
                    loss=loss1+loss2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Pretrain epoch [{}/{}],loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
            # torch.save({
            #     'ae_state_dict': self.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            # },'./预训练模型/'+time+'.pth.tar')
     def fit(self,x1,x2,x,y, lr=0.02,batch_size=3762, num_epochs=60,dim_subspace= 16, ro=8.0,alpha=0.04,save_dir=""):#10X数据参数,也可64
            y-=1
            Y = torch.tensor(y).long()
            X = torch.tensor(x)
            X1 = torch.tensor(x1)
            X2 = torch.tensor(x2)


            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.001, last_epoch=-1, verbose=False)#学习率衰减
            C = self.self_expression.Coefficient.detach().to('cpu')
            self.y_pred,_ = utils.post_proC(C.detach().numpy(), n_clusters, dim_subspace, ro)
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
            acc = np.round(utils.acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing spectral_clustering: AMI= %.4f,ACC= %.4f, NMI= %.4f, ARI= %.4f' % (ami, acc, nmi, ari))
            
            num = X.shape[0]
            num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
       
            np.random.seed(0)
            
            for epoch in range(num_epochs):
                self.eval()
                C = self.self_expression.Coefficient.detach().to('cpu').numpy()
                self.y_pred,_ = utils.post_proC(C, n_clusters, dim_subspace, ro)
                ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                acc = np.round(utils.acc(y, self.y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score (y, self.y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                print('spectral_clustering: AMI= %.4f,ACC= %.4f, NMI= %.4f, ARI= %.4f' % (ami, acc, nmi, ari))
            

                train_loss = 0.0
                recon_loss_val = 0.0
                if epoch % 10:
                    self.train()
                    for batch_idx in range(num_batch):
                        xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                        x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]   
                        x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]         
                        optimizer.zero_grad()
                        inputs = Variable(xbatch)
                        inputs1 = Variable(x1batch)
                        inputs2 = Variable(x2batch)
                        z, zc,x1_rec, x2_rec= self.forward(inputs)
                        loss,loss_ae,tracelossx,loss_selfExp,penalty,loss_s=self.loss_fn(x=xbatch,x1=x1batch,x2=x2batch, x1_rec=x1_rec , x2_rec=x2_rec,z=z, zc=zc, weight_ae=0.5, weight_trace=1, weight_selfExp=0.01,weight_C=1.0)
                        loss.backward()                                                                            
                        optimizer.step()
                        sch.step()
                   
                        print("#Epoch %3d: loss: %.4f , loss_ae: %.4f, tracelossx: %.4f, loss_selfExp: %.4f,penalty:%.4f,loss_s:%.4f" % (
                        epoch + 1, loss/num , loss_ae/num ,tracelossx,loss_selfExp/num,penalty/num,loss_s/num))
            # self.save_checkpoint({'epoch': epoch+1,
            #         'state_dict': self.state_dict(),
            #         'y_pred': self.y_pred,
            #         'y': y
            #         }, epoch+1, filename=save_dir)
            return self.y_pred, ami, acc, nmi, ari

if __name__ == '__main__':
    x1, x2, x ,y= load_data()
    z1_dim = 32
    z2_dim = 32
    z_dim = 32
    n=3762
    X1_dim =2000
    X2_dim =49
    X_dim = X1_dim+X2_dim
    n_clusters=16
    N =128
    M=64
    epochs = 100 
    nb_classes = len(set(y))    
    msc = MSCNet(X1_dim,X2_dim,X_dim,z_dim,N,M,n,nb_classes)
    print('预训练自编码器')
    msc.pretrian(x1,x2,x)
    y_pred, ami,acc, nmi, ari = msc.fit(x1 ,x2, x ,y,save_dir='./数据模型')


