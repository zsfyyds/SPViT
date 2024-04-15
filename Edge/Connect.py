import pickle
import socket
import time

import torch
import models as Vgg
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import random
import numpy as np
import os

import matplotlib


# ====================================================================================================
#                                  Soekct Program
# ====================================================================================================

# 创建一个socket套接字，该套接字还没有建立连接
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定监听端口，这里必须填本机的IP192.168.27.238，localhost和127.0.0.1是本机之间的进程通信使用的
server.bind(('192.168.137.1', 6688))
# 开始监听，并设置最大连接数
server.listen(5)

print(u'waiting for connect...')
# 等待连接，一旦有客户端连接后，返回一个建立了连接后的套接字和连接的客户端的IP和端口元组
connect, (host, port) = server.accept()
print(u'the client %s:%s has connected.' % (host, port))


class Data(object):

    def __init__(self, dfx_client,gradient_start_time):
        self.dfx_client=dfx_client
        self.gradient_start_time=gradient_start_time



def sendData(dfx_client,gradient_start_time):
    data=Data(dfx_client,gradient_start_time)

    # **序列化**——就是把内存里面的这些对象给变成一连串的字节描述的过程。
    # 序列化：把对象转换为字节序列的过程称为对象的序列化。
    str=pickle.dumps(data)

    # 把str转换成长度为length的字节字符串，并发送
    connect.send(len(str).to_bytes(length=6, byteorder='big'))
    connect.send(str)


def receiveData():
            lengthData=connect.recv(6)
            if lengthData==b'quit':
                return lengthData
            length=int.from_bytes(lengthData, byteorder='big')
            b=bytes()
            count=0
            while True:
                value=connect.recv(length)
                b=b+value
                count+=len(value)
                if count>=length:
                    break

            # 反序列化：把字节序列恢复为对象的过程称为对象的反序列化。
            data=pickle.loads(b)
            return data

# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 7) #7 is my numbr of classes
net_glob_server = Vgg.VggServerNetwork()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)


lr = 0.0001



# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
criterion = nn.CrossEntropyLoss()

# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user  # idx_collect是什么

    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)  # fx_client客户端部分的模型的输出
    y = y.to(device)

    # ---------forward prop-------------
    fx_server = net_glob_server(fx_client)  # net_glob_server服务器端模型

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    #acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()  # 客户端的梯度不是由服务器端产生的，而是由客户端自己的输出产生的
    optimizer_server.step()

    return dfx_client


# =====================================================================================================

while True:
    fx =receiveData()
    if fx==b'quit':
        print("training completed!")
        break
    client_fx, y, l_epoch_count, l_epoch, idx, len_batch,inter_start_time=fx.client_fx,fx.labels,fx.iter,fx.slocal_ep,fx.idx,fx.len_batch,fx.inter_start_time

    if inter_start_time:
        inter_end_time = time.time()
        print('intermedia data is ',inter_end_time-inter_start_time)

    server_start_train=time.time()
    dfx=train_server(client_fx, y, l_epoch_count, l_epoch, idx, len_batch)
    server_end_train=time.time()
    print('server train is ',server_end_train-server_start_train)

    if inter_start_time:
        gradient_start_time = time.time()
    else:
        gradient_start_time=None
    sendData(dfx,gradient_start_time)
    # fx_client=np.frombuffer(data)
    # fx_client=torch.from_numpy(fx_client)
#     dfx=train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch)
# connect.sendall(dfx)