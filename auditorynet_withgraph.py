import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from einops import rearrange
import time
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import random
from fairseq.modules.multihead_attention import MultiheadAttention
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
'''
初始版本，各层之间无FFN，crossattention无残差连接。
'''

class localneuralattention(nn.Module):
    def __init__(self,windowsize,dim,head):
        super(localneuralattention,self).__init__()
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=head)
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(dim, eps = 1e-5)
        self.windowsize = windowsize
    def forward(self,x):
        '''
        输入格式(b,t,f)
        处理过程：
        (b*w,t',f)
        输出格式(b,t,f)
        '''
        n1 = x.shape[1] // self.windowsize
        x1 = rearrange(x,'b (n t1) f -> (b n) t1 f',n = n1)
        x1 = x1.permute([1,0,2])#(t1 (b n) f)
        attn, attn_weights = self.mha(query = x1, key = x1, value = x1)
        # attn = attn.permute([1,0,2])
        x2 = attn.permute([1,0,2])
        x2 = rearrange(x2, '(b n) t1 f-> b (n t1) f', n = n1)
        x1 = self.ln(x + x2)
        return x1


class crossneuralattention(nn.Module):
    '''
    通用连接方式
    '''
    def __init__(self, dim, head):
        super(crossneuralattention, self).__init__()
        self.mha = MultiheadAttention(embed_dim = dim, num_heads = head)
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(dim, eps = 1e-5)
        self.alpha = nn.Parameter(torch.ones(1)*0.5,requires_grad = True)
    def forward(self,x1,x2):
        '''
        x1:当前层
        x2:上一层
        '''
        x3 = x1 * self.alpha + x2 * (1-self.alpha)
        x1 = x1.permute([1,0,2])
        x3 = x3.permute([1,0,2])
        x,attn_weights = self.mha(query = x1, key = x3, value = x3)
        x = x.permute([1,0,2])
        return x

class crossneuralattention2(nn.Module):
    '''
    专用连接方式
    '''
    def __init__(self, dim, head):
        super(crossneuralattention2, self).__init__()
        self.mha = MultiheadAttention(embed_dim = dim, num_heads = head)
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(dim, eps = 1e-5)
        self.fc = nn.Linear(dim * 2, 2)
    def forward(self,x1,x2):
        '''
        x1:当前层
        x2:上一层
        '''
        # x3 = x1 * self.alpha + x2 * (1-self.alpha)
        x11 = torch.mean(x1, dim = 1)
        x22 = torch.mean(x2, dim = 1)
        weight = torch.cat([x11,x22],dim = -1)
        weight = self.fc(weight)
        weight = self.softmax(weight)
        x3 = x1 * weight[:,0].unsqueeze(-1).unsqueeze(-1) + x2 * weight[:,1].unsqueeze(-1).unsqueeze(-1)
        x1 = x1.permute([1,0,2])
        x3 = x3.permute([1,0,2])
        x,attn_weights = self.mha(query = x1, key = x3, value = x3)
        x = x.permute([1,0,2])
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

class classifier(nn.Module):
    def __init__(self, dim, classnum):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, dim // 4)
        self.fc3 = nn.Linear(dim // 4, classnum)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        # x = torch.mean(x, dim = -2)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.dropout(self.relu(x))
        x = self.fc3(x)
        return x

class AuditoryNet(nn.Module):
    def __init__(self,dim, head, ffn_dim, classnum, hidden_channels):
        super(AuditoryNet, self).__init__()
        self.la1 = localneuralattention(windowsize = 3, dim = dim, head = head)
        self.la2 = localneuralattention(windowsize = 9, dim = dim, head = head)
        self.la3 = localneuralattention(windowsize = 27, dim = dim, head = head)
        self.la4 = localneuralattention(windowsize = 324, dim = dim, head = head)
        self.ffn = FeedForwardNetwork(dim = dim, ffn_dim = ffn_dim)
        self.classifier = classifier(dim, classnum)
        self.conv1 = GCNConv(dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(dim, eps = 1e-5)
        self.ln2 = nn.LayerNorm(dim, eps = 1e-5)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index):
        x = self.ln1(x)
        # print(x.shape)
        x = rearrange(x, '(b n) t f -> b n t f', n = 4)
        x[:, 0, :, :] = self.la1(x[:, 0, :, :])
        x[:, 1, :, :] = self.la2(x[:, 1, :, :])
        x[:, 2, :, :] = self.la3(x[:, 2, :, :])
        x[:, 3, :, :] = self.la4(x[:, 3, :, :])
        x = rearrange(x, 'b n t f -> (b n) t f', n = 4)
        x = torch.mean(x, dim = 1)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training = self.training)
        # x = self.conv2(x, edge_index)
        # x = self.relu(x)
        # x = F.dropout(x, p = 0.5, training = self.training)
        # out = self.classifier(x[:,3,:])
        x = rearrange(x, '(b n) f -> b n f', n = 4)
        out = self.classifier(x[:,3,:])
        return out