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
from torch.nn.parameter import Parameter
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

def custom_constraint(x):
    return torch.clamp(x, 0, 1)

class simhgcl(nn.Module):
    def __init__(self, args, tgt_type, tgt_num=None, feat_keys=None, data_size=None):
        super(simhgcl, self).__init__()
        self.feat_keys = feat_keys
        self.num_layers = 0
        for k in self.feat_keys:
            self.num_layers = max(self.num_layers, len(k)-1)
        self.data_size = data_size
        self.tgt_num = tgt_num
        self.encoder = SeHGNN(args=args, feat_keys=self.feat_keys, tgt_type=tgt_type, data_size=self.data_size)
        num_channels = len(self.feat_keys)
        self.v = 1
        self.v_pre = 1
        self.proj_head = nn.Sequential(nn.Linear(args.hidden*(self.num_layers+num_channels-1), args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden))
        #self.proj_head = nn.Sequential(nn.Linear(args.hidden*(num_channels), args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden))
        self.init_emb()

    def init_emb(self):
        #initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def _protect(self, x1, x2, prt_adj=None, prt_local=0.05):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        #sim_matrix = torch.exp(sim_matrix / tau)
        
        neg_o = sim_matrix.detach().clone()
        neg_o[range(batch_size), range(batch_size)] = 0
        neg_o = torch.softmax(neg_o, dim = 1)
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        delta = sim_matrix-pos_sim
        neg_i = sim_matrix.detach().clone()
        neg_i[delta>=0] = 1
        neg_i = torch.softmax(neg_i, dim = 1)
        KL = torch.sum(neg_i * (neg_i.log() - neg_o.log()), dim=1)
        prt_nums = int(prt_local * prt_adj.shape[0])
        _, prt_index = torch.topk(KL, prt_nums, largest=False)
        prt_adj[prt_index] = 0
        return prt_adj

    def _radius(self, x1, x2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        delta = sim_matrix-pos_sim
        with torch.no_grad():
            self.v_pre = self.v
            var = delta.var(dim=1, unbiased=True)
            self.v = var.mean()

    def gen_ran_input(self, feats):
        adv_feats = {}
        eta = args.eta
        if self.v_pre != 1:
            eta = eta * (self.v / (self.v_pre))
        #print(eta)
        for k in feats.keys():
            adv_feats[k] = torch.zeros_like(feats[k])
            #adj_std = feats[k].std(dim = 1, keepdim = True).repeat(1, feats[k].shape[-1])
            #noise = torch.normal(0,adj_std).to(device)
            noise = torch.normal(0,torch.ones_like(feats[k])*args.eta).to(device)
            adv_feats[k] = feats[k] + noise
        return adv_feats
    
    def forward(self, feats):
        adv_feats = self.gen_ran_input(feats)
        x = self.encoder(feats)
        y = self.proj_head(x)
        adv_x = self.encoder(adv_feats) 
        adv_y = self.proj_head(adv_x)
        #self._radius(y, adv_y)
        return y, adv_y

    def loss_cal(self, x1, x2):
    
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1)-pos_sim)
        loss1 = - torch.log(loss1).mean()
        
        return loss1

def gen_ran_output(data, model, vice_model, args, labels=None):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head' or (len(param.data)==1) or name.split('.')[1] == 'embeding' or name.split('.')[1] == 'feature_projection' or name.split('.')[1] == 'fc_after_concat':
        #if name.split('.')[0] == 'proj_head' or (len(param.data)==1):
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)   
    _, z2 = vice_model(data)
    if labels is not None:
        emb, y = vice_model.encoder.get_embeddings(data, labels)
        return emb, y
    return z2

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
        prt_masks[node_type[i]] = torch.ones(v).to(device)
        num_nodes[node_type[i]] = v
        
    new_adjs = {}
    perturbation = {}
    all_perturbation = 0
    for et in edge_type:
        new_adjs[et] = adjs[et]
        if args.scen == 'rem':
            perturbation[et] = new_adjs[et].to_dense().nonzero().shape[0] * args.att_local
        elif args.scen == 'add_rem':
            perturbation[et] = (new_adjs[et].to_dense().shape[0]*new_adjs[et].to_dense().shape[1]) * args.att_local
        else:
            raise ValueError('threat_model not set correctly.')
        perturbation[et] = int(perturbation[et])

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

def neighbor_aggregation(g, adjs, edge_type, num_hops, tgt_type, prt_adj=None):
    print(f'Current num hops = {num_hops} for neighbor propagation')
    prop_tic = datetime.datetime.now()

    raw_feats = hg_propagate_feat(g, adjs, edge_type, tgt_type, num_hops, prt_adj)

    #print(f'For target type {tgt_type}, feature keys (num={len(raw_feats)}):', end='')
    #print()

    data_size = {k: v.size(-1) for k, v in raw_feats.items()}

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    '''
    for k, v in raw_feats.items():
        print('feats: ', k, v.size(), v[:,-1].max(), v[:,-1].mean())
    #input()
    '''
    
    return raw_feats, data_size

def sim(x, y):
    x_abs = x.norm(dim=1)
    y_abs = y.norm(dim=1)
    sim_score = torch.einsum('ik,jk->ij', x, y) / torch.einsum('i,j->ij', x_abs, y_abs)
    return sim_score

def hg_propagate_feat(g, adjs, edge_type, tgt_type, num_hops, prt_adj=None):
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
                        if (i >= 1) and prt_adj is not None:
                            if k2[0] == tgt_type and k1[-1] != tgt_type:
                                prt_index = (prt_adj==0).nonzero()
                                for k in prt_index:
                                    nei = adj[k].reshape(-1)
                                    nei = (nei>0).nonzero().reshape(-1)
                                    nei_x = last_layer[k1][nei]
                                    if nei_x.shape[0] > 0:
                                        max_pool,_ = torch.max(nei_x, dim=0)
                                        layer_feat[k2+k1[1:]][k] = max_pool
                            if k2[0] == tgt_type and k1[-1] == tgt_type:
                                prt_index = (prt_adj==0).nonzero()
                                for k in prt_index:
                                    nei = adj[k].reshape(-1)
                                    nei = (nei>0).nonzero().reshape(-1)
                                    nei_x = last_layer[k1][nei]
                                    x = raw_feats[tgt_type][k]
                                    sim_score = sim(x, nei_x)
                                    if sim_score.shape[-1] > 0:
                                        _,max_index = torch.max(sim_score, dim=1)
                                        layer_feat[k2+k1[1:]][k] = nei_x[max_index]
        
        for k in layer_feat.keys():
            if (k[0] == tgt_type):
                raw_feats[k] = layer_feat[k]
        layer_feats.append(layer_feat)
    return raw_feats

def worst_margin_compute(feats, vice_feats):
    f_abs = feats.norm(dim=1)
    vf_abs = vice_feats.norm(dim=1)
    between_sim = torch.einsum('ik,jk->ij', feats, vice_feats) / torch.einsum('i,j->ij', f_abs, vf_abs)
    #between_sim = torch.exp(between_sim / tau)
    pos_sim = torch.diag(between_sim)
    margin = pos_sim - between_sim
    worst_margin, index = torch.min(margin, dim = 1)
    var = (between_sim-pos_sim).var(unbiased=True)
    return worst_margin, var.mean()



def train(model, raw_feats, labels, num_channels, path, trainval_nid, test_nid, type='PRE'):

    all_loss = []
    margins = []
    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_m = -1e9
    
    starttime = datetime.datetime.now()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_index, val_index, test_index = dataset_spilt(trainval_nid, test_nid)
    

    for e in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        feats, vice_feats = model(raw_feats)
        worst_margin, delta = worst_margin_compute(vice_feats, feats)    
        #loss = model.loss_cal(feats, vice_feats)
        loss = model.loss_cal(vice_feats, feats)
        robust_num = (worst_margin>=0).sum()
        avg_margin = worst_margin.mean().detach().cpu()*1000
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss: {:.4f}, robust num: {}'.\
                format(e, loss.item(), robust_num) )
        
        #print('Epoch: {}, Loss: {:.4f}'.\
        #        format(e, loss.item()) )
        
        best_m = max(best_m, avg_margin)
        
        if loss.item() < best:
            best = loss.item()
            best_t = e
            cnt_wait = 0
            torch.save(model.state_dict(), path+'Seed_'+str(args.seed)+'_'+type+'_clean_model.pkl')
        else:
            cnt_wait += 1
            
        all_loss.append(loss.item())
        margins.append(avg_margin)
        '''
        if e == args.epoch-1 or cnt_wait == args.patience:
            model.load_state_dict(torch.load(path+'Seed_'+args.seed+'_'+type+'_clean_model.pkl'))
            model.eval()
            #raw_feats = model.gen_ran_input(raw_feats)
            emb, y = model.encoder.get_embeddings(raw_feats, labels)
            evaluate_embedding(emb, emb, y, model.num_layers+num_channels-1, args, device, trainval_nid, test_nid)
            #evaluate_embedding(emb, y, num_channels, args, device, trainval_nid, test_nid)
            break
        '''
    #os.remove(path+type+'_clean_model.pkl')
    #return model
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
        adjs[k].storage._value = torch.ones(adjs[k].nnz())
    adjs, tgt_type, edge_type, prt_masks, perturbation, num_nodes, in_dims = preprocess(args, g, adjs, dl)
    tgt_num = num_nodes[tgt_type]
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)
    
    cur_adjs = {}
    
    ### protect
    budget = {}
    all_budget = 0
    for k in adjs.keys():
        cur_adjs[k] = adjs[k].to_dense().to(device)
        budget[k] = int(cur_adjs[k].nonzero().shape[0] * args.prt_local)
        all_budget += budget[k]
    
    raw_feats, data_size = neighbor_aggregation(g, cur_adjs, edge_type, args.num_hops, tgt_type)
    feat_keys = raw_feats.keys() 
    print('feat_keys: {}'.format(feat_keys))
    pre_model = simhgcl(args=args, tgt_type=tgt_type, tgt_num=tgt_num, feat_keys=feat_keys, data_size=data_size).to(device)
    if os.path.exists(checkpt_folder+'Seed_'+str(args.seed)+'_PRE_clean_model.pkl') is False:
        train(pre_model, raw_feats, labels, len(feat_keys), checkpt_folder, trainval_nid, test_nid)
    pre_model.load_state_dict(torch.load(checkpt_folder+'Seed_'+str(args.seed)+'_PRE_clean_model.pkl'))

    feats, vice_feats = pre_model(raw_feats)
    prt_adj = pre_model._protect(feats, vice_feats, prt_masks[tgt_type], args.prt_local)
    print('-----------Protect Successfully!-----------')
   
    ### pretrain attack       
    '''
    torch.set_printoptions(profile="full")
    if args.attack_load:
        if args.att_local != 0:
            with open(log_folder + 'supervised_meta_attack_{}_{}_adjs.pkl'.format(args.att_local, args.att_lr), 'rb') as f:
                cur_changing = pkl.load(f)
            for k in adjs.keys():
                cur_adjs[k] = cur_adjs[k] + cur_changing[k].to_dense().detach().to(device)
    '''
    ###
    raw_feats, data_size = neighbor_aggregation(g, cur_adjs, edge_type, args.num_hops, tgt_type, prt_adj)
    feat_keys = raw_feats.keys() 
    print('feat_keys: {}'.format(feat_keys))
    prt_model = simhgcl(args=args, tgt_type=tgt_type, tgt_num=tgt_num, feat_keys=feat_keys, data_size=data_size).to(device)
    if os.path.exists(checkpt_folder+'Seed_'+str(args.seed)+'_PRT_clean_model.pkl') is False:
        train(prt_model, raw_feats, labels, len(feat_keys), checkpt_folder, trainval_nid, test_nid, 'PRT')
    prt_model.load_state_dict(torch.load(checkpt_folder+'Seed_'+str(args.seed)+'_PRT_clean_model.pkl'))

    prt_model.eval()
    emb, y = prt_model.encoder.get_embeddings(raw_feats, labels)
    ### test attack 
    if args.attack_load:
        if args.att_local != 0:
            with open(log_folder + 'supervised_meta_attack_{}_{}_adjs.pkl'.format(args.att_local, args.att_lr), 'rb') as f:
                cur_changing = pkl.load(f)
            for k in adjs.keys():
                cur_adjs[k] = cur_adjs[k] + cur_changing[k].to_dense().detach().to(device)
    atk_feats, _ = neighbor_aggregation(g, cur_adjs, edge_type, args.num_hops, tgt_type, prt_adj)
    atk_emb, y = prt_model.encoder.get_embeddings(atk_feats, labels)
    ### 
    #evaluate_embedding(emb, emb, y, prt_model.num_layers+len(feat_keys)-1, args, device, trainval_nid, test_nid)  ### pretrain phase
    evaluate_embedding(emb, atk_emb, y, prt_model.num_layers+len(feat_keys)-1, args, device, trainval_nid, test_nid) ### test phase


if __name__ == '__main__':

    if args.dataset == 'ACM':
        # We do not keep the node type `field`(F) as raw features represent exactly the distribution of nearby field nodes
        args.ACM_keep_F = False

    #for k,v in args.__dict__.items():
    #    print(f"{k}: {v}")
    print(args)
    main(args)