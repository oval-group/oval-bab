# Scaling the Convex Barrier with Sparse Dual Algorithms

This folder contains scripts to aid towards the replication of the findings described in: [Scaling the Convex Barrier with Sparse Dual Algorithms](https://arxiv.org/abs/2101.05844).
Note, however, that these were based on an earlier version of the codebase.
If you use them in your research, please cite:

```
@Article{DePalma2024,
    title={Scaling the Convex Barrier with Sparse Dual Algorithms},
    author={De Palma, Alessandro and Behl, Harkirat Singh and Bunel, Rudy and Torr, Philip H. S. and Kumar, M. Pawan},
    journal={Journal of Machine Learning Research},
    year={2024}
}
```

## Installation

Please follow the instructions provided in the main README file of this repository. 
A Gurobi installation is required for some of the experiments. 

## Employed models

All of the networks used in the experiments, except the fully-connected network from Figure 4, are provided within the `models` folder.
The mentioned network can be trained using the code at https://github.com/alessandrodepalma/ibpr/, as follows:
```bash
python train/train_main.py \
--train_mode train --dataset cifar10 --net fcshallow_7000 \
--train_batch 100 --test_batch 100 --train_eps 0.00784313725 --start_eps_factor 1.1 --anneal \
--train_att_n_steps 8 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.035 \
--opt sgd --lr 1e-2 --lr_factor 0.95 --cont_lr_decay --cont_lr_mix  --mix --mix_epochs 600 --n_epochs 800 \
--l1_reg 1e-5 --test_freq 50 --do_not_ver --random_seed 0
```
and then convert it into `.onnx` for easy loading and verification. For instance, this is done when running the following:
```bash
python verify/certify.py \
--dataset cifar10 --net fcshallow_7000 --test_eps 0.00784313725 --ib_batch_size 2000 \
--use_oval_bab --oval_bab_config ./bab_configs/colt_models.json --oval_bab_timeout 1800 --load_model path_to_model
```
The model then needs to be placed at `./models/onnx/cifar_fcshallow_7000_2_255_adv.onnx`. 

## Running the experiments

The paper's experiments can be replicated as follows (adapting hardware parameters according to one's needs):

```bash
# complete verification -- In the paper, Gurobi uses 6 CPU cores
python scripts_jmlr/run_anderson_bab_cifar.py --gpu_id 0 --cpus 0-5
# incomplete verification -- the paper uses 4 CPU cores
python scripts_jmlr/run_anderson_incomplete.py --gpu_id 0 --cpus 0-3 --experiment all
```

and then plotted as:
```bash
# complete verification
python scripts_jmlr/plot_verification.py
# incomplete verification. 
# pandas tables from which table 4 can be generated will be in the `~/results/eran_convsmall` folder.
python scripts_jmlr/parse_bounds.py