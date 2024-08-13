import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import gc
import datetime


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion in SeHGNN.
    '''
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection in SeHGNN.
    '''
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


unfold_nested_list = lambda x: sum(x, [])

class SeHGNN(nn.Module):
    '''
        The SeHGNN model.
    '''
    def __init__(self, args, feat_keys, tgt_type, data_size=None, num_heads=1):
        super(SeHGNN, self).__init__()
        self.dataset = args.dataset
        
        nfeat = args.embed_size
        hidden = args.hidden
        dropout = args.dropout

        self.feat_keys = sorted(feat_keys)
        self.num_channels = num_channels = len(self.feat_keys)
        self.tgt_type = tgt_type
        self.residual = args.residual

        self.num_layers = 0
        for k in self.feat_keys:
            self.num_layers = max(self.num_layers, len(k)-1)

        self.input_drop = nn.Dropout(args.input_drop)

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, nfeat))

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(nfeat, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(hidden, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(args.n_fp_layers - 1)])
            )
        )

        self.semantic_fusion_all = Transformer(hidden, num_heads=num_heads, att_drop=args.att_drop, act=args.act)
        self.semantic_fusion = nn.ModuleList()
        for l in range(self.num_layers):
            self.semantic_fusion.append(Transformer(hidden, num_heads=num_heads, att_drop=args.att_drop, act=args.act))
        
        self.bn = torch.nn.BatchNorm1d((self.num_layers+num_channels-1) * hidden)
        #self.bn = torch.nn.BatchNorm1d((num_channels) * hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden)

        self.alpha = nn.Parameter(torch.tensor([0.]))

        self.reset_parameters()

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
        nn.init.constant_(self.alpha, 0.1)

    def get_embeddings(self, feats, labels):
        ret = []
        y = []

        with torch.no_grad():
            x= self.forward(feats)
            ret.append(x.cpu().numpy())
                
            label = labels.clone()
            y.append(label.cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)

        return ret, y

    def forward(self, feature_dict):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}

        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            # Freebase has so many metapaths that we use feature projection per target node type instead of per metapath
            features = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}

        else:
            assert 0

        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels
        D = features[self.tgt_type].shape[1]

        x = [features[k] for k in self.feat_keys]
        x = torch.stack(x, dim=1) # [B, C, D]
        x = self.feature_projection(x)
        
        h = [x[:, 0, :].reshape(B, 1, D) for l in range(self.num_layers)]
        cnt = 0
        for k in self.feat_keys:
            if len(k) != 1:
                h[len(k)-2] = torch.cat((h[len(k)-2], x[:, cnt, :].reshape(B, 1, D)), dim = 1)
            cnt += 1    
        for l in range(self.num_layers):
            h[l] = self.semantic_fusion[l](h[l], mask=None)
            if l != 0:
                #h[0] = torch.cat((h[0], (self.alpha ** l) * h[l]), dim = 1)
                h[0] = torch.cat((h[0], h[l]), dim = 1)
        #x = self.semantic_fusion_all(h[0], mask=None).transpose(1,2)
        #x = self.bn(torch.cat((x.reshape(B, -1), h[0].reshape(B, -1)), dim = -1))
        x = self.bn(h[0].reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])
        return x


class SeHGNN_CLS(nn.Module):
    '''
        The SeHGNN model.
    '''
    def __init__(self, args, feat_keys, tgt_type, data_size=None, num_heads=1):
        super(SeHGNN_CLS, self).__init__()
        self.dataset = args.dataset
        
        self.device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'

        nfeat = args.embed_size
        hidden = args.hidden
        dropout = args.dropout
        n_task_layers = args.n_task_layers
        nclass = args.num_classes

        self.feat_keys = sorted(feat_keys)

        self.num_layers = 0
        for k in self.feat_keys:
            self.num_layers = max(self.num_layers, len(k)-1)
            
        self.num_channels = num_channels = len(self.feat_keys)
        self.tgt_type = tgt_type
        self.residual = args.residual

        self.input_drop = nn.Dropout(args.input_drop)

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, nfeat))

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(nfeat, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(hidden, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(args.n_fp_layers - 1)])
            )
        )

        self.semantic_fusion_all = Transformer(hidden, num_heads=num_heads, att_drop=args.att_drop, act=args.act)
        self.semantic_fusion = nn.ModuleList()
        for l in range(self.num_layers):
            self.semantic_fusion.append(Transformer(hidden, num_heads=num_heads, att_drop=args.att_drop, act=args.act))
        
        #self.fc_after_concat = nn.Linear(num_channels * hidden, hidden)
        self.fc_after_concat = nn.Linear((self.num_layers+num_channels-1+num_channels) * hidden, hidden)
        
        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden)
            
        if self.dataset not in ['IMDB', 'Freebase']:
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

    def forward(self, feature_dict):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}

        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            # Freebase has so many metapaths that we use feature projection per target node type instead of per metapath
            features = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}

        else:
            assert 0

        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels
        D = features[self.tgt_type].shape[1]

        x = [features[k] for k in self.feat_keys]
        x = torch.stack(x, dim=1) # [B, C, D]
        x = self.feature_projection(x)

        h = [x[:, 0, :].reshape(B, 1, D) for l in range(self.num_layers)]
        cnt = 0
        for k in self.feat_keys:
            if len(k) != 1:
                h[len(k)-2] = torch.cat((h[len(k)-2], x[:, cnt, :].reshape(B, 1, D)), dim = 1)
            cnt += 1    
        
        for l in range(self.num_layers):
            h[l] = self.semantic_fusion[l](h[l], mask=None).transpose(1,2)
            if l != 0:
                h[0] = torch.cat((h[0], h[l]), dim = -1)

        x = self.semantic_fusion_all(x, mask=None).transpose(1,2)

        x = self.fc_after_concat(torch.cat((x.reshape(B, -1), h[0].reshape(B, -1)), dim = -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])
        if self.dataset not in ['IMDB', 'Freebase']:
            return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x
