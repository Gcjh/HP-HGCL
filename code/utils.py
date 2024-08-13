import os
import sys
import gc
import random

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score

sys.path.append('../data')
from data_loader import data_loader
from loader import loader

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

def set_random_seed(args):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'
    return device

def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')

def flip_edges(adjs, fragiles):
    adjs_flipped = adjs.copy()
    for k in adjs.keys():
        adj_flipped[k] = adjs[k] - fragiles[k]
    return adjs_flipped


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def dataset_spilt(trainval_nid, test_nid, val_ratio = 0.2):
    train_nid = trainval_nid.copy()
    np.random.shuffle(train_nid)

    split = int(train_nid.shape[0]*val_ratio)

    val_nid = train_nid[:split]
    train_nid = train_nid[split:]
    train_nid = np.sort(train_nid)
    val_nid = np.sort(val_nid)

    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    print(f'#Train {train_node_nums}, #Val {valid_node_nums}, #Test {test_node_nums}')

    return train_nid, val_nid, test_nid

unfold_nested_list = lambda x: sum(x, [])

class LogReg(nn.Module):
    def __init__(self, args, num_channels=1):
        super(LogReg, self).__init__()
        self.dataset =  args.dataset
        hidden = args.hidden*num_channels
        dropout = args.dropout
        n_task_layers = args.n_task_layers
        nclass = args.num_classes

        if self.dataset not in ['IMDB', 'Freebase']:
            '''
            self.task_mlp = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(dropout),]
                + unfold_nested_list([[
                   nn.Linear(hidden, hidden),
                   nn.BatchNorm1d(hidden, affine=False),
                   nn.PReLU(),
                   nn.Dropout(dropout),] for _ in range(n_task_layers - 1)])
                + [nn.Linear(hidden, nclass),
                   nn.BatchNorm1d(nclass, affine=False, track_running_stats=False)]
                )
            )
            '''
            self.fc = nn.Linear(hidden, nclass)
            for m in self.modules():
                self.weights_init(m)
            #'''    
        else:
            self.task_mlp = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(dropout))]
                + [nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden, affine=False),
                    nn.PReLU(),
                    nn.Dropout(dropout)) for _ in range(n_task_layers - 1)]
                + [nn.Sequential(
                    nn.Linear(hidden, nclass),
                    nn.LayerNorm(nclass, elementwise_affine=False),
                    )]
            )
            self.reset_parameters()
            
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()
    def forward(self, x):
        if self.dataset not in ['IMDB', 'Freebase']:
            ret = self.fc(x)
            return ret
            #return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x


def svc_classify(x, y, train_index, test_index, search):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    if search:
        params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
        classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
    else:
        classifier = SVC(C=10)
    classifier.fit(x_train, y_train)
    
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    mif = f1_score(gt_test, preds_test, average='micro')
    maf = f1_score(gt_test, preds_test, average='macro')

    return accuracy, mif, maf


def logistic_classify(x, atk_x, y, num_channels,args, device,  trainval_nid, test_nid):
    
    nb_classes = np.unique(y).shape[0]
    if args.dataset == 'IMDB':
        xent = nn.BCEWithLogitsLoss()
    else:
        xent = nn.CrossEntropyLoss()
    
    hid_units = x.shape[1]

    mif = []
    maf = []
    
    mif_val = []
    maf_val = []
    
    train_index, val_index, test_index = dataset_spilt(trainval_nid, test_nid)

    train_embs, val_embs, test_embs = x[train_index], x[val_index], atk_x[test_index]
    train_lbls, val_lbls, test_lbls= y[train_index], y[val_index], y[test_index]

    train_embs, train_lbls = torch.from_numpy(train_embs).to(device), torch.from_numpy(train_lbls).to(device)
    val_embs, val_lbls= torch.from_numpy(val_embs).to(device), torch.from_numpy(val_lbls).to(device)
    test_embs, test_lbls= torch.from_numpy(test_embs).to(device), torch.from_numpy(test_lbls).to(device)
   
    for _ in tqdm(range(args.max_iter)):
        #log = LogReg(args)
        log = LogReg(args, num_channels)
        log.to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01)

        best_val = 1e9
        
        for it in range(200):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
        
            log.eval()
            # val
            logits = log(val_embs)
            loss_val = xent(logits, val_lbls)
            if args.dataset != 'IMDB':
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0.).int()
        
            gt_val = val_lbls.cpu().squeeze()
            preds_val = preds.cpu().squeeze()
        
            # test
            logits = log(test_embs)

            if args.dataset != 'IMDB':
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (logits > 0.).int()
        
            gt_test = test_lbls.cpu().squeeze()
            preds_test = preds.cpu().squeeze()
            
            if loss_val < best_val:
                best_val = loss_val
                mif_val.append( f1_score(gt_val, preds_val, average='micro') )
                maf_val.append( f1_score(gt_val, preds_val, average='macro') )
                mif.append( f1_score(gt_test, preds_test, average='micro') )
                maf.append( f1_score(gt_test, preds_test, average='macro') )
        
    index = np.argmax(maf_val)

    #return np.mean(accs_val), np.mean(mif_val), np.mean(maf_val), np.mean(accs), np.mean(mif), np.mean(maf)
    return mif_val[index], maf_val[index], mif[index], maf[index]


def evaluate_embedding(embeddings, atk_embeddings, labels, num_channels, args, device, trainval_nid, test_nid, search=True):
    
    x, y = np.array(embeddings), np.array(labels)
    atk_x = np.array(atk_embeddings)
        
    mif_val, maf_val, mif, maf = logistic_classify(x,atk_x,y,num_channels,args,device, trainval_nid, test_nid)
    
    print('Val_micro:{:.2f}, Val_macro:{:.2f}'.format(mif_val*100, maf_val*100))
    print('Test_micro:{:.2f}, Test_macro:{:.2f}'.format(mif*100, maf*100))
    #return acc_val, acc
    
    
def load_dataset(args):
    #test = loader(f'{args.root}/{args.dataset}')
    dl = data_loader(f'{args.root}/{args.dataset}')

    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))

    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    trainval_nid = np.nonzero(dl.labels_train['mask'])[0]
    test_nid = np.nonzero(dl.labels_test['mask'])[0]

    init_labels[trainval_nid] = dl.labels_train['data'][trainval_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)
    init_labels = torch.LongTensor(init_labels)

    # === adjs ===
    # print(dl.nodes['attr'])
    # for k, v in dl.nodes['attr'].items():
    #     if v is None: print('none')
    #     else: print(v.shape)
    adjs = [] if args.dataset != 'Freebase' else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset == 'Freebase':
            name = f'{dst_type_idx}{src_type_idx}'
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            #print(adj)

    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        # for i, etype in enumerate(g.etypes):
        #     src, dst, eid = g._graph.edges(i)
        #     adj = SparseTensor(row=dst.long(), col=src.long())
        #     print(etype, adj)

        # g.ndata['feat']['A'] = A # not work
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list

        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1) # each movie has single director

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 5:
            g.nodes['K'].data['K'] = K
    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]
        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        g = dgl.heterograph(new_edges)

        g.nodes['P'].data['P'] = P # [3025, 1902]
        g.nodes['A'].data['A'] = A # [5959, 1902]
        g.nodes['C'].data['C'] = C # [56, 1902]
        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K # [1902, 1902]
    elif args.dataset == 'Freebase':
        # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
        #  1: 19427  all <-- 1
        #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
        #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
        #  4: 17641  4 <-- all
        #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
        #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
        #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7
        for i in range(8):
            kk = str(i)
            print(f'==={kk}===')
            for k, v in adjs.items():
                t, s = k
                assert s == t or f'{s}{t}' not in adjs
                if s == kk or t == kk:
                    if s == t:
                        print(k, v.sizes(), v.nnz(),
                              f'symmetric {v.is_symmetric()}; selfloop-ratio: {v.get_diag().sum()}/{v.size(0)}')
                    else:
                        print(k, v.sizes(), v.nnz())

        adjs['00'] = adjs['00'].to_symmetric()
        g = None
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        if args.ACM_keep_F:
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP, 'PK': PK, 'KP': KP}
        else:
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    elif args.dataset == 'Freebase':
        new_adjs = {}
        for rtype, adj in adjs.items():
            dtype, stype = rtype
            if dtype != stype:
                new_name = f'{stype}{dtype}'
                assert new_name not in adjs
                new_adjs[new_name] = adj.t()
        adjs.update(new_adjs)
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
