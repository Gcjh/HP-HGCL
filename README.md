# HP-HGCL-Master
This project involves the code and supplementary materials of paper "Improving Heterogeneous Contrastive Learning Robustness via Self-Supervised Hierarchical Protection".



## Dependencies
* pytorch == 1.12.0
* numpy == 1.24.2
* scikit-learn == 1.2.1
* tqdm == 4.64.1
* scipy == 1.11.3
* seaborn == 0.13.0
* networkx == 3.0

Install other dependencies:

```setup
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```

## Data

* HGB datasets for node classification

These datasets include four medium-scale datasets. Please download them (`DBLP.zip`, `ACM.zip`, `IMDB.zip`) from [HGB repository](https://github.com/THUDM/HGB) and extract content under the folder `'./data/'`.

## Run
Running HP-HGCL is followed as:

```
IMDB
    python new_main.py --epoch 100 --dataset IMDB --num-hops 4 --hidden 512 --embed-size 512 --patience 20 --eta 0.01 --att_local 0.1 --prt_local 0.2 --seed 0
```   

```
DBLP
    python new_main.py --epoch 100 --dataset DBLP --num-hops 4 --hidden 512 --embed-size 512 --patience 20 --eta 1 --att_local 0.1 --prt_local 0.2 --seed 0
```

```
ACM
    python new_main.py --epoch 100 --dataset ACM --num-hops 3 --hidden 512 --embed-size 512 --patience 20 --eta 1 --att_local 0.1 --prt_local 0.05 --seed 0
```

To reproduce the results of HP-HGCL, please run the above commands.

