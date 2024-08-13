import gc
import datetime
import torch

def neighbor_aggregation(g, adjs, edge_type, num_hops, tgt_type):
    print(f'Current num hops = {num_hops} for neighbor propagation')
    prop_tic = datetime.datetime.now()

    raw_feats = hg_propagate_feat(g, adjs, edge_type, tgt_type, num_hops)

    print(f'For target type {tgt_type}, feature keys (num={len(raw_feats)}):', end='')
    print()
    for k, v in raw_feats.items():
        print(k, v.size())
    
    data_size = {k: v.size(-1) for k, v in raw_feats.items()}

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    return raw_feats, data_size
    
def hg_propagate_feat(g, cur_adjs, edge_type, tgt_type, num_hops):
    raw_feats = {}
    if num_hops <= 2:
        raw_feats[tgt_type] = g.nodes[tgt_type].data[tgt_type]
        for k in cur_adjs.keys():
            if (k.find(tgt_type) == 0) and (k in edge_type):
                to = g.nodes[k[1]].data[k[1]]
                raw_feats[k] = torch.matmul(cur_adjs[k], to).requires_grad_()
            else:
                continue
        if num_hops == 1:
            return raw_feats
        to = g.nodes[tgt_type].data[tgt_type]
        for k in cur_adjs.keys():
            if (k.find(tgt_type) == 1) and (k in edge_type):
                raw_feats[k[1]+k[0]+k[1]] = torch.matmul(torch.matmul(cur_adjs[k[1]+k[0]], cur_adjs[k]), to).requires_grad_()
            else:
                continue   
    else:
        raise ValueError('num_hops not set correctly.')
    return raw_feats