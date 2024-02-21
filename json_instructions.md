# Json Instructions

This file contains some instructions on how to write a `.json` configuration file for the branch and bound framework. 
We will use `bab_configs/oval21_vnncomp21.json` as an example file.
Further information can be found in the parsing algorithm, implemented within `bab_from_json` of `tools/bab_tools/bab_runner.py`.

## Bounding algorithms
The bounding algorithms to be stratified (see section 6.2 from https://arxiv.org/pdf/2101.05844.pdf) are passed as a 
list of dictionaries, in order of increasing computational cost, using the `nets` entry of the `bounding` key.

`bounding_algorithm` can be either `prox`, to use the proximal solver from [Bunel et al. 2020b](https://arxiv.org/abs/2002.10410), 
`propagation` for linear bound propagation methods (including a beta-CROWN [Wang et al. 2021](https://arxiv.org/abs/2103.06624) implementation),
`dual-anderson` for algorithms on the convex hull of the ReLU with the preceding linear layer, from [De Palma et al. 2024](https://arxiv.org/abs/2101.05844).

`batch_size` indicates the largest number of subproblems to split at once. Twice at many will be bounded upon. Reduce this in case of memory errors.
`auto_iters` automatically increases the number of iterations of a bounding algorithm, as described in https://arxiv.org/abs/2104.06718.

The `params` defined within each dictionary refer to the hyper-parameters of each bounding algorithms, 
which are used in the `solver.py` files (see main README).

## Intermediate bounds
`ibs` accepts a dictionary for the bounding algorithm to be used for pre-activation bounds (under `loose_ib`), and whether these bounds should be kept fixed during the procedure (`fixed_ib`). 
Keep `joint_ib` (only supported by `propagation`) equal to false for the best performance under our implementation.

## Upper bounds
`upper_bounding` refers to the parameters of the adversarial attack being employed. Only MI-FGSM (`mi_fgsm`) is currently supported.

## Branching
`branching` refers to the branching strategy. For instance `FSB` is the strategy from https://arxiv.org/abs/2104.06718, which accepts a bounding algorithm to do the filtering. I would keep it as shown in the example (best bounds between the method from Wong and Kolter and CROWN). 
`SR` would run the strategy from https://arxiv.org/abs/1909.06588.
Other options can be found within the initilizer of `BranchingChoice` in `plnn/branch_and_bound/branching_scores.py`.