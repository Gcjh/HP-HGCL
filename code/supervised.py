import os
import gc
import time
import uuid
import argparse
import random
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

from arguments import parse_args
from model import *
from utils import *
from sparse_tools import SparseAdjList
from scipy.sparse import csr_matrix 
from scipy.optimize import linear_sum_assignment

from torch.utils.tensorboard import SummaryWriter   
import pickle as pkl

args = parse_args()
device = set_random_seed(args)
tau = args.tau

def SparseTensor_T(adj):
    return SparseTensor(row=adj.storage.col(), col=adj.storage.row(), sparse_sizes=(adj.size(1), adj.size(0)))

def SparseTensor_del_node(adj, t, c=0):
    if c:
        index = (adj.storage.col() == t).nonzero()
    else:
        index = (adj.storage.row() == t).nonzero()
    new_row = adj.storage.row().cpu().numpy()
    new_row = torch.from_numpy(np.delete(new_row, index))
    new_col = adj.storage.col().cpu().numpy()
    new_col = torch.from_numpy(np.delete(new_col, index))
    return SparseTensor(row=new_row, col=new_col, sparse_sizes=(adj.size(1), adj.size(0)))

def preprocess(args, g, adjs, dl):
    tgt_type = '0'
    node_type = []
    in_dims = {}
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        node_type = ['A', 'P', 'T', 'V']
        edge_type = ['AP', 'PA', 'PT', 'PV', 'TP', 'VP']
        in_dims = {'A':334, 'P':4231, 'T': 50, 'V':20}
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        if args.ACM_keep_F:
            node_type = ['P', 'A', 'C', 'K']
            edge_type = ['PP', 'PA', 'AP', 'PC', 'CP', 'PK', 'KP']
        else:
            node_type = ['P', 'A', 'C']
            edge_type = ['PP', 'PA', 'AP', 'PC', 'CP']
        in_dims = {'P':1902, 'A':1902, 'C': 1902}
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        if args.num_hops > 5:
            node_type = ['M', 'D', 'A', 'K']
            edge_type = ['MD', 'DM', 'MA', 'AM', 'MK', 'KM']
        else:
            node_type = ['M', 'D', 'A']
            edge_type = ['MD', 'DM', 'MA', 'AM']
        in_dims = {'M':3489, 'D':3341, 'A':3341}
    elif args.dataset == 'Freebase':
        tgt_type = '0'
        ### TODO:: Freebase.framework
        return 0
    else:
        assert 0

    prt_masks = {}
    num_nodes = {}
    for i, (k,v) in enumerate(dl.nodes['count'].items()):
        if i == len(node_type):
            break
        prt_masks[node_type[i]] = torch.ones(v).requires_grad_().to(device)
        num_nodes[node_type[i]] = v
        
    new_adjs = {}
    perturbation = {}
    all_perturbation = 0
    for et in edge_type:
        new_adjs[et] = adjs[et]
        '''
        if args.scen == 'rem':
            perturbation[et] = new_adjs[et].to_dense().nonzero().shape[0] * args.att_local
        elif args.scen == 'add_rem':
            perturbation[et] = (new_adjs[et].to_dense().shape[0]*new_adjs[et].to_dense().shape[1]) * args.att_local
        else:
            raise ValueError('threat_model not set correctly.')
        '''
        perturbation[et] = int(new_adjs[et].to_dense().nonzero().shape[0] * args.att_local)

    return new_adjs, tgt_type, edge_type, prt_masks, perturbation, num_nodes, in_dims

def get_argmaxPos(x):
    index = torch.argmax(x)
    row = index // x.shape[1]
    col = index % x.shape[1]
    return row, col
    
def get_prt(x, prt_masks, k):
    if prt_masks[k][x] == 0:    
        return 1
    return 0

def neighbor_aggregation(g, adjs, edge_type, num_hops, tgt_type):
    print(f'Current num hops = {num_hops} for neighbor propagation')
    prop_tic = datetime.datetime.now()

    raw_feats = hg_propagate_feat(g, adjs, edge_type, tgt_type, num_hops)

    print(f'For target type {tgt_type}, feature keys (num={len(raw_feats)}):', end='')
    print()

    data_size = {k: v.size(-1) for k, v in raw_feats.items()}

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    '''
    for k, v in raw_feats.items():
        print('feats: ', k, v.size(), v[:,-1].max(), v[:,-1].mean())
    input()
    '''
    
    return raw_feats, data_size
    
def hg_propagate_feat(g, adjs, edge_type, tgt_type, num_hops):
    raw_feats = {}
    layer_feats = []
    raw_feats[tgt_type] = g.nodes[tgt_type].data[tgt_type].to(device)
    for i in range(num_hops):
        layer_feat = {}
        if i == 0:
            for k in adjs.keys():
                if k in edge_type:
                    to = g.nodes[k[-1]].data[k[-1]].to(device)
                    adj = adjs[k].to_dense().to(device)
                    s = adj.sum(dim = 1, keepdim = True).repeat(1, adj.shape[-1])
                    s[s==0] = 1
                    adj = adj / s
                    layer_feat[k] = torch.matmul(adj, to)
                    del s, adj, to
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            last_layer = layer_feats[-1]
            for k1 in last_layer.keys():
                for k2 in adjs.keys():
                    if (k2[-1] == k1[0]) and (k2 in edge_type):
                        adj = adjs[k2].to_dense().to(device)
                        s = adj.sum(dim = 1, keepdim = True).repeat(1, adj.shape[-1])
                        s[s==0] = 1
                        adj = adj / s
                        layer_feat[k2+k1[1:]] = torch.matmul(adj, last_layer[k1])
                        del s, adj
                        gc.collect()
                        torch.cuda.empty_cache()
        for k in layer_feat.keys():
            if (k[0] == tgt_type):
                raw_feats[k] = layer_feat[k]
        layer_feats.append(layer_feat)
        del layer_feat
        gc.collect()
        torch.cuda.empty_cache()
    del layer_feats
    gc.collect()
    torch.cuda.empty_cache()
    return raw_feats


def attack_iteration(g, adjs, labels, tgt_type, edge_type, perturbation, prt_masks, trainval_nid, test_nid):
    
    adj_changing={}
    count = {}
    all_perturbation = 0
    all_count = 0
    train_index, val_index, test_index = dataset_spilt(trainval_nid, test_nid)
    for k in adjs.keys():
        adj_changing[k] = torch.zeros_like(adjs[k].to_dense()).to(device)
        count[k] = 0
        all_perturbation += perturbation[k]
    
    while all_count < all_perturbation:
        mp_fragile = {}
    
        for k in adjs.keys():
            adj_changing[k] = adj_changing[k].requires_grad_()
            adj_changing[k].retain_grad()
            mp_fragile[k] = adjs[k].to_dense().to(device).detach() + adj_changing[k]
        
        atk_feats, atk_size = neighbor_aggregation(g, mp_fragile, edge_type, args.num_hops, tgt_type)
        atk_keys = atk_feats.keys()  
        
        atk_model = SeHGNN_CLS(args=args, feat_keys=atk_keys, tgt_type=tgt_type, data_size=atk_size).to(device)
        opt = torch.optim.Adam(atk_model.parameters(), lr=args.att_lr)
    
        if args.dataset == 'IMDB':
            xent = nn.BCEWithLogitsLoss()
        else:
            xent = nn.CrossEntropyLoss()
        meta_grad = {}
        
        for it in tqdm(range(args.attack_iter), desc='attack'):
            atk_model.train()
            opt.zero_grad()
            feats = atk_model(atk_feats)
            atk_loss = xent(feats[train_index], labels[train_index])
            atk_loss.backward(retain_graph=True)
            opt.step()
            for k in mp_fragile.keys():
                if adj_changing[k].grad is not None:
                    if it:
                        meta_grad[k] = meta_grad[k] + adj_changing[k].grad.clone().detach()
                    else:
                        meta_grad[k] = adj_changing[k].grad.clone().detach()
                    
        if all_count == 0:
            ori_perturbation = all_perturbation
            for k in adjs.keys():
                if k not in meta_grad.keys():
                    all_perturbation -= perturbation[k]
            print('Attack Num change from {} to {}'.format(ori_perturbation, all_perturbation))
        
        atk_model.eval()
        for k in adjs.keys():
            x = mp_fragile[k].clone().detach()
            y = adj_changing[k].clone().detach()
            del mp_fragile[k], adj_changing[k]
            gc.collect()
            torch.cuda.empty_cache()
            mp_fragile[k] = x
            adj_changing[k] = y
            
        Score = {}
        for k in meta_grad.keys():
            Score[k] = torch.mul(meta_grad[k], 1-2*mp_fragile[k])
            Score[k] = Score[k] - torch.min(Score[k])
            
        for k in Score.keys():
            if count[k] < perturbation[k]:
                x, y = get_argmaxPos(Score[k])
                if adj_changing[k][x][y] != 0:
                    while adj_changing[k][x][y] != 0:
                        Score[k][x][y] = 0
                        if ((k[1]+k[0]) in Score.keys()):
                            Score[k[1]+k[0]][y][x] = 0
                        x, y = get_argmaxPos(Score[k])
                adj_changing[k][x][y] = 1-2*mp_fragile[k][x][y]
                adj_changing[k[1]+k[0]][y][x] = 1-2*mp_fragile[k[1]+k[0]][y][x]
                all_count+=1
                count[k]+=1
            else:
                continue
        print('Attack Num {}, Current Attack {}'.format(all_perturbation, all_count))
        
        del atk_model, opt, meta_grad, Score, atk_feats, mp_fragile
        gc.collect()
        torch.cuda.empty_cache()
               
    return adj_changing



def worst_margin_compute(feats, vice_feats):
    
    f_abs = feats.norm(dim=1)
    vf_abs = vice_feats.norm(dim=1)
    
    between_sim = torch.einsum('ik,jk->ij', feats, vice_feats) / torch.einsum('i,j->ij', f_abs, vf_abs)
    pos_sim = between_sim.diag()
    
    margin = pos_sim - between_sim.t() 
    
    worst_margin, index = torch.min(margin, dim = 0)
    
    return worst_margin, margin.mean(dim = 0)

def train(model, raw_feats, labels, path, log_folder, trainval_nid, test_nid):

    all_loss = []
    margins = []
    cnt_wait = 0
    best = 1e9
    best_t = 0
    train_index, val_index, test_index = dataset_spilt(trainval_nid, test_nid)
    
    starttime = datetime.datetime.now()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.dataset == 'IMDB':
        xent = nn.BCEWithLogitsLoss()
    else:
        xent = nn.CrossEntropyLoss()
    
    for e in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        feats = model(raw_feats)
        loss = xent(feats[train_index], labels[train_index])
        loss.backward()
        optimizer.step()
        model.eval()
        feats_val = feats[val_index]
        l = xent(feats_val, labels[val_index]).item()
        print('Epoch: {}, Train_Loss: {:.4f}, Val_Loss: {:.4f}'.format(e, loss.item(), l) )
        if l < best:
            best = l
            best_t = e
            cnt_wait = 0
            torch.save(model.state_dict(), path+str(starttime)+'_clean_model.pkl')
        else:
            cnt_wait += 1            
        if (e+1 == args.epoch) or (cnt_wait == args.patience):
            model.load_state_dict(torch.load(path+str(starttime)+'_clean_model.pkl'))
            model.eval()
            os.remove(path+str(starttime)+'_clean_model.pkl')
            # test
            feats = model(raw_feats)
            feats_test = feats[test_index]
            test_lbls = labels[test_index]
            if args.dataset != 'IMDB':
                preds = torch.argmax(feats_test, dim=1)
            else:
                preds = (feats_test > 0.).int()
        
            gt_test = test_lbls.cpu().squeeze()
            preds_test = preds.cpu().squeeze()
            mi = f1_score(gt_test, preds_test, average='micro')
            ma = f1_score(gt_test, preds_test, average='macro')
            #worst_margin, margin = worst_margin_compute(preds_test, gt_test)    
            #robust_num = (worst_margin>0).sum()
            #avg_margin = margin.mean().detach().cpu()
            print('Test Results: micro:{:.2f}, macro:{:.2f}'.format(mi*100, ma*100))
            break
    #return all_loss, margins

def main(args):

    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    log_folder = f'./output/{args.dataset}/log/'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    path = f'../data/{args.dataset}/'
    
    g, adjs, labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)
        
    if args.dataset != 'IMDB':
        labels = labels.long().to(device)
    else:
        labels = labels.float().to(device)
        
    args.num_classes = num_classes
    
    for k in adjs.keys():
        adjs[k].storage._value = None
        #adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
        adjs[k].storage._value = torch.ones(adjs[k].nnz())
    
    adjs, tgt_type, edge_type, prt_masks, perturbation, num_nodes, in_dims = preprocess(args, g, adjs, dl)
    
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)
    
    cur_adjs = {}    
    ### attack
    #'''    
    cur_changing = attack_iteration(g, adjs, labels, tgt_type, edge_type, perturbation, prt_masks, trainval_nid, test_nid)
    sparse_changing = {}
    for k in cur_changing.keys():
        adj = cur_changing[k].detach().cpu()
        index = adj.nonzero().T
        row = index[0]
        col = index[1]
        s_adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(adj.shape[0], adj.shape[1]))
        s_adj.storage._value = torch.ones(s_adj.nnz())
        sparse_changing[k] = s_adj
    with open(log_folder + 'supervised_meta_attack_{}_{}_adjs.pkl'.format(args.att_local, args.att_lr), 'wb') as f:
        pkl.dump(sparse_changing, f)
    #'''
    
    '''
    if args.attack_load:
        with open(log_folder + 'supervised_meta_attack_{}_{}_adjs.pkl'.format(args.att_local, args.att_lr), 'rb') as f:
        #with open('meta_attack_adj_IMDB_0.25.pkl', 'rb') as f:
            cur_changing = pkl.load(f)
    '''
    
    ### protect
    '''
    if args.prt_load:
        with open(log_folder + 'prt_0.1_{}_adjs.pkl'.format(args.prt_local), 'rb') as f:
            prt_masks = pkl.load(f) 
    '''
    
    ### attack after protect    
    '''
    if args.attack_load:
        with open(log_folder + 'meta_attack_prt_{}_{}_adjs.pkl'.format(args.att_local, args.prt_local), 'rb') as f:
            cur_changing = pkl.load(f)
    '''
    
    
    for k in adjs.keys():
        #cur_adjs[k] = adjs[k].to_dense().to(device)
        cur_adjs[k] = adjs[k].to_dense().to(device) + cur_changing[k].to_dense().detach().to(device)
        #cur_control = torch.outer(prt_masks[k[0]], prt_masks[k[1]]).to(device)
        #cur_adjs[k] = (adjs[k].to_dense().to(device) + torch.mul(cur_changing[k].to(device), cur_control)).detach()  
    
    raw_feats, data_size = neighbor_aggregation(g, cur_adjs, edge_type, args.num_hops, tgt_type)
    feat_keys = raw_feats.keys() 
    model = SeHGNN_CLS(args=args, feat_keys=feat_keys, tgt_type=tgt_type, data_size=data_size).to(device)
    train(model, raw_feats, labels, checkpt_folder, log_folder, trainval_nid, test_nid)


if __name__ == '__main__':

    if args.dataset == 'ACM':
        # We do not keep the node type `field`(F) as raw features represent exactly the distribution of nearby field nodes
        args.ACM_keep_F = False

    #for k,v in args.__dict__.items():
    #    print(f"{k}: {v}")
    print(args)
    main(args)