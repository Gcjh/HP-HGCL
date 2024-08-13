import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SSRP')
    ## For environment costruction
    parser.add_argument('--seed', type=int, default=1,
                        help='the seed used in the training')
    parser.add_argument('--dataset', type=str, default='ACM',
                        choices=['DBLP', 'ACM', 'IMDB', 'Freebase'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--root', type=str, default='../data/')

    ##------SeHGNN arguments------
    ## SeHGNN: For network structure
    parser.add_argument('--num-hops', type=int, default=2,
                        help='number of hops for propagation of raw labels')
    parser.add_argument('--n-fp-layers', type=int, default=1,
                        help='the number of mlp layers for feature projection')
    parser.add_argument('--n-task-layers', type=int, default=1,
                        help='the number of mlp layers for the downstream task')
    parser.add_argument('--embed-size', type=int, default=512,
                        help='inital embedding size of nodes with no attributes')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout on activation')
    parser.add_argument('--input-drop', type=float, default=0.1,
                        help='input dropout of input features')
    parser.add_argument('--att-drop', type=float, default=0.,
                        help='attention dropout of model')
    parser.add_argument('--act', type=str, default='none',
                        choices=['none', 'relu', 'leaky_relu', 'sigmoid'],
                        help='the activation function of the transformer part')
    parser.add_argument('--residual', action='store_true', default=False,
                        help='whether to add residual branch the raw input features')
    ## SeHGNN: for training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='whether to amp to accelerate training with float16(half) calculation')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=20,
                        help='early stop patience')
    ##------SimHGCL arguments------
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--eta', type=float, default=1.0, help='0.1, 1.0, 10, 100, 1000')
    parser.add_argument('--epoch', type=int, default=100, help='Maxinum number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--max_iter', type=int, default=1, help='the number of contrastive learning iteration')
    parser.add_argument('--ratio', type=int, default=20, help='ratio.')

    ##------Defend arguments------
    parser.add_argument('--scen', choices=['rem','add_rem'], default='rem', help='Scenarios of surrogate attack model')
    parser.add_argument('--att_local', type=float, default=0.05, help='the local budget of surrogate model and robustness certification')
    parser.add_argument('--att_lr', type=float, default=0.001, help='the local budget of surrogate model and robustness certification')
    parser.add_argument('--attack_iter', type=int, default=20, help='the number of contrastive learning iteration')
    parser.add_argument('--attack_load', default=True, help='whether load the previous attack')
    parser.add_argument('--prt_local', type=float, default=0.03, help='the local budget of immunization')
    parser.add_argument('--prt_load', default=True, help='whether load the previous protection')

    return parser.parse_args(args)

