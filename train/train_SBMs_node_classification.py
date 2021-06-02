"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):  # eg. batch_graphs: a batch of graphs can be viewed as a large graph that have many disjoint connected components.     batch_labels: torch.Size([total_num_nodes_in_batch])
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat  # torch.Size([total_num_nodes_in_batch])
        batch_e = batch_graphs.edata['feat'].to(device)  # torch.Size([total_num_edges_in_batch, 1]) 다 1로 채워져 있음 
        batch_labels = batch_labels.to(device)           # torch.Size([total_num_nodes_in_batch])
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)  # torch.Size([total_num_nodes_in_batch, pos_enc_dim])
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)  # torch.rand:  uniform distribution on the interval [0, 1)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0) # TODO: 1/2의 확률로 pos_emb의 부호를 바꾸는 이유가 뭐지?
        except:
            batch_lap_pos_enc = None
            
        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)  # batch_scores: [num_nodes_in_batch, 2]
    
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()  # iter반복하면서 epoch_loss에 계속 더함
        epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


