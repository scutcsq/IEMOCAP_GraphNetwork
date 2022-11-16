import os
# from turtle import forward
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import librosa
import time
import random
from sklearn import metrics
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.backends import cudnn
# from statistics import mode
import matplotlib.pyplot as plt
# from dynamic_window_mask2 import DynamicTransformer
# from OriginalTransformer import OriginalTransformer3
# from dynamicwindow18 import DynamicTransformer1,DynamicTransformer2,DynamicTransformer3
# from xiaorong import DynamicTransformer1
import torch.nn.functional as F
# from transformer import Vanilla_Transformer
# from hierarchicaltransformer import localTranformer,localTranformer2
import math
import lmdb
from auditorynet_withgraph import AuditoryNet

torch.set_num_threads(1)
gen_data = False
voting = True

#-----------------------------------------------限制随机------------------------------------------

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark= False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
#-------------------------------------------------生成DataLoader-----------------------------------
class FCModel(nn.Module):
    def __init__(self):
        super(FCModel,self).__init__()
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = torch.mean(x,dim = 1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.dropout(self.relu(x))
        x = self.fc3(x)
        return x
class ADDataset(Dataset):
    def __init__(self,data,label):
        self.data = np.load(data)
        self.label = np.load(label)
    def __getitem__(self,item):
        return torch.Tensor(self.data[item,:,:]),torch.Tensor(self.label[item,:])
    def __len__(self):
        return len(self.label)

# class lmdb_dataset(Dataset):
#     def __init__(self,out_path,mode):
#         self.env = lmdb.open(out_path + mode)
#         self.txn = self.env.begin(write = False)
#         self.len = self.txn.stat()['entries']
#     def __getitem__(self,index):
#         key_data = 'data-%05d' %index
#         key_label = 'label-%05d' %index

#         data = np.frombuffer(self.txn.get(key_data.encode()),dtype = np.float32)
#         data = torch.FloatTensor(data.reshape(-1,1024).copy())
#         label = np.frombuffer(self.txn.get(key_label.encode()),dtype = np.int64)
#         label = torch.LongTensor(label.copy()).squeeze()

#         return data, label
#     def __len__(self):
#         return int(self.len / 2)

class lmdb_dataset(Dataset):
    def __init__(self,out_path,mode):
        self.env = lmdb.open(out_path + mode)
        self.txn = self.env.begin(write = False)
        self.len = self.txn.stat()['entries']
    def __getitem__(self,index):
        key_data = 'data-%05d' %index
        key_label = 'label-%05d' %index
        # key_mask = 'mask-%05d' %index

        data = np.frombuffer(self.txn.get(key_data.encode()),dtype = np.float32)
        data = torch.FloatTensor(data.reshape(-1,1024).copy())
        label = np.frombuffer(self.txn.get(key_label.encode()),dtype = np.int64)
        label = torch.LongTensor(label.copy()).squeeze()
        # mask = np.frombuffer(self.txn.get(key_mask.encode()),dtype = np.float32)
        # mask = torch.FloatTensor(mask.copy())

        return data, label
    def __len__(self):
        return int(self.len / 2)

class lmdb_dataset2(Dataset):
    def __init__(self,out_path,mode):
        self.env = lmdb.open(out_path + mode)
        self.txn = self.env.begin(write = False)
        self.len = self.txn.stat()['entries']
    def __getitem__(self,index):
        key_data = 'data-%05d' %index
        key_label = 'label-%05d' %index
        key_mask = 'mask-%05d' %index

        data = np.frombuffer(self.txn.get(key_data.encode()),dtype = np.float32)
        data = torch.FloatTensor(data.reshape(-1,1024).copy())
        label = np.frombuffer(self.txn.get(key_label.encode()),dtype = np.int64)
        label = torch.LongTensor(label.copy()).squeeze()
        mask = np.frombuffer(self.txn.get(key_mask.encode()),dtype = np.float32)
        mask = torch.FloatTensor(mask.copy())

        return data, label,mask
    def __len__(self):
        return int(self.len / 3)

class IEMOCAPDataset(Dataset):
    def __init__(self, out_path, mode):
        super(IEMOCAPDataset,self).__init__()
        self.env = lmdb.open(out_path + mode)
        self.txn = self.env.begin(write = False)
        self.len = self.txn.stat()['entries']

    def __getitem__(self,idx):
        key_data = 'data-%05d' %idx
        key_label = 'label-%05d' %idx

        data = np.frombuffer(self.txn.get(key_data.encode()),dtype = np.float32)
        data = torch.FloatTensor(data.reshape(-1,1024).copy())
        # data = torch.FloatTensor(data.copy())
        label = np.frombuffer(self.txn.get(key_label.encode()),dtype = np.int64)
        label = torch.LongTensor(label.copy()).squeeze()
        data2 = torch.cat([data.unsqueeze(0), data.unsqueeze(0), data.unsqueeze(0), data.unsqueeze(0)],dim = 0)
        edge_index = torch.tensor([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]],dtype = torch.long)
        graph_data = Data(x = data2, y = label, edge_index = edge_index)
        return graph_data
    def __len__(self):
        return int(self.len/3)
#--------------------------------------------------数据加载---------------------------------
final_wa = []
final_ua = []
for i in range(1,6):
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()

    # traindata = r'train'+str(i)+'data.npy'
    # trainlabel = r'train'+str(i)+'label.npy'

    # developdata = r'valid'+str(i)+'data.npy'
    # developlabel = r'valid'+str(i)+'label.npy'

    # train_dataset = ADDataset(traindata,trainlabel)#training set
    # develop_dataset = ADDataset(developdata,developlabel)#valid set
    # out_path = r'../IEMOCAP/IEMOCAP_full_release/new_database_wavlm_mask_324/'
    out_path = r'/148Dataset/data-chen.shuaiqi/IEMOCAP/IEMOCAP_full_release/new_database_wavlm_mask_324/'
    # out_path = r'./new_database_hubert_mask_324/'
    trainplace = r'train' + str(i)
    validplace = r'valid' + str(i)
    train_dataset = IEMOCAPDataset(out_path,trainplace)
    develop_dataset = IEMOCAPDataset(out_path,validplace)
    trainDataset = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,drop_last = False)
    developDataset = DataLoader(dataset=develop_dataset,batch_size=32,shuffle= False)

    #---------------------------------------------------参数设置---------------------------------

    # model = Adaptive_token_Block(324,12,4,1024,1024,8,512).to(device)
    # model = DynamicTransformer(1024,8,512,4).to(device)
    # model = FCModel().to(device)
    # model = OriginalTransformer3(classnum = 4,feadim = 1024,n_head = 8,ffndim = 512).to(device)
    # model = Vanilla_Transformer(input_dim = 1024, ffn_embed_dim = 512, num_layers = 7, num_heads = 8, num_classes = 4,dropout = 0.3).to(device)
    # model = DynamicTransformer1(feadim = 1024, n_head = 8, FFNdim = 512, classnum = 4).to(device)
    # model = localTranformer2(feadim = 1024,n_head = 8, FFNdim = 512, classnum = 4).to(device)
    # model = haveatry().to(device)
    model = AuditoryNet(dim = 1024, head = 8, ffn_dim = 512, classnum = 4, hidden_channels = 1024).to(device)
    WD = 5e-4
    LR_DECAY = 0.5
    EPOCH = 120
    STEP_SIZE = 5
    lr = 1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr = 5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)
    optimizer = torch.optim.SGD(model.parameters(),lr = lr, momentum = 0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120, 1e-4 * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 3,T_mult = 2, eta_min = 1e-4 * 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 3, T_mult = 2)
    loss = nn.CrossEntropyLoss().to(device)
    #--------------------------------------------------train--------------------------------------------
    best_wa = 0
    best_ua = 0
    num = 0
    for epoch in range(EPOCH):
        model.train()
        loss_tr = 0.0
        start_time = time.time()
        pred_all,actu_all = [],[]
        for step, datas in enumerate(trainDataset, 0):
            datas = datas.to(device)
            optimizer.zero_grad()
            out = model(datas.x, datas.edge_index)
            err1 = loss(out,datas.y)
            err1.backward()
            optimizer.step()
            pred = torch.max(out.cpu().data, 1)[1].numpy()
            actu = datas.y.cpu().data.numpy()
            pred_all += list(pred)
            actu_all += list(actu)
            loss_tr += err1.cpu().item()
        loss_tr = loss_tr / len(trainDataset.dataset)
        pred_all, actu_all = np.array(pred_all), np.array(actu_all)
        wa_tr = metrics.accuracy_score(actu_all, pred_all)
        ua_tr = metrics.recall_score(actu_all, pred_all,average='macro')
        end_time = time.time()
        print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)
        print('所耗时长:',str(end_time-start_time),'s')
        scheduler.step()
        # torch.save(model.state_dict(), 'result2/'+str(epoch)+'.pkl')
    # #---------------------------------------------------develop-----------------------------------------
        model.eval()
        loss_de = 0.0
        start_time = time.time()
        pred_all,actu_all = [],[]
        for step, datas in enumerate(developDataset, 0):
            datas = datas.to(device)
            #原有
            with torch.no_grad():
                out = model(datas.x, datas.edge_index)
            err1 = loss(out, datas.y)
            pred = torch.max(out.cpu().data, 1)[1].numpy()
            actu = datas.y.cpu().data.numpy()
            pred_all += list(pred)
            actu_all += list(actu)
            loss_de += err1.cpu().item()
        loss_de = loss_de / len(developDataset.dataset)
        pred_all, actu_all = np.array(pred_all,dtype=int), np.array(actu_all,dtype=int)

        wa_de = metrics.accuracy_score(actu_all, pred_all)
        ua_de = metrics.recall_score(actu_all, pred_all,average='macro')
        if (ua_de+wa_de) >= (best_ua + best_wa):
            best_ua = ua_de
            best_wa = wa_de
            num = epoch
        end_time = time.time()
        print('VALID:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)
        print('所耗时长:  ',str(end_time-start_time),'s')
    final_wa.append(best_wa)
    final_ua.append(best_ua)
    print('当前折最好结果: | wa: %.3f' %best_wa, '|ua: %.3f' %best_ua)
final_wa = np.array(final_wa)
final_ua = np.array(final_ua)
final_wa1 = np.mean(final_wa)
final_ua1 = np.mean(final_ua)
print('五折交叉验证结果:  | wa: %.3f' %final_wa1,'|ua: %.3f' %final_ua1)