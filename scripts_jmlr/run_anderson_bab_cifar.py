import os
import argparse


def cifar_oval_runner_command(gpu_id, cpus, timeout, pdprops, nn, alg_specs, batch_size, max_solver_batch):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python ./tools/bab_tools/bab_runner.py " \
              f"--timeout {timeout} --pdprops {pdprops} --dataset cifar_oval --nn_name {nn}  {alg_specs} " \
              f"--batch_size {batch_size} --max_solver_batch {max_solver_batch}" \
              " --bheuristic SR --looser_ib"
    return command


def iclr_experiments(gpu_id, cpus, n_splits=1, modulo_do=0, use_autostrat=False):
    # n_splits and modulo_do determine how many blocks to split the commands to, and which part to do
    timeout = 3600
    pdprops_list = ["base_100.pkl", "wide_100.pkl", "deep_100.pkl"]
    nn_names = ["cifar_base_kw", "cifar_wide_kw", "cifar_deep_kw"]
    max_solver_batch_list = [25000, 17000, 17000]
    batch_size_list = [150, 100, 100]

    algo_dict = {
        "prox": "--method prox --tot_iter 100 --eta 1e2 --feta 1e2 --parent_init",
        "bigm-adam": "--method bigm-adam --tot_iter 180 --init_step 1e-2 --fin_step 1e-4 --parent_init",
        "cut": "--method cut --init_step 1e-3 --fin_step 1e-6 --tot_iter 180 --hard_iter 100 --cut_add 2 "
               "--dualinit_init_step 1e-2 --dualinit_fin_step 1e-4 --parent_init",
        "gurobi-anderson": "--method gurobi-anderson --gurobi_p 6 --n_cuts 1",
        "anderson-mip": "--method anderson-mip --gurobi_p 6",
    }
    algo_dict["cut_no_easy"] = algo_dict["cut"] + " --no_easy"

    if use_autostrat:
        algo_dict["cut"] = algo_dict["cut"] + " --hard_overhead 6"
        algo_dict["gurobi-anderson"] = algo_dict["gurobi-anderson"] + " --hard_overhead 20"

    command_list = []
    for pdprops, nn, batch_size, solver_batch in zip(pdprops_list, nn_names, batch_size_list, max_solver_batch_list):
        for algo_specs in algo_dict.values():
            command_list.append(
                cifar_oval_runner_command(gpu_id, cpus, timeout, pdprops, nn, algo_specs, batch_size, solver_batch))

    command_list = [command_list[0]]
    for idx, ccommand in enumerate(command_list):
        if n_splits > 1 and idx % n_splits != modulo_do:
            continue
        print(ccommand)
        os.system(ccommand)


def jmlr_experiments(gpu_id, cpus, n_splits=1, modulo_do=0, use_autostrat=False):
    # n_splits and modulo_do determine how many blocks to split the commands to, and which part to do
    timeout = 3600
    pdprops_list = ["base_100.pkl", "wide_100.pkl", "deep_100.pkl"]
    nn_names = ["cifar_base_kw", "cifar_wide_kw", "cifar_deep_kw"]
    max_solver_batch_list = [25000, 17000, 17000]
    batch_size_list = [150, 100, 100]

    cut_specs = "--method cut --init_step 1e-3 --fin_step 1e-6 --tot_iter 180 --hard_iter {} --cut_add 2 " \
                "--dualinit_init_step 1e-2 --dualinit_fin_step 1e-4 --parent_init"
    spfw_specs = "--method sp-fw --tot_iter 180 --hard_iter {} --fw_start 10 --dualinit_init_step 1e-3 " \
                 "--dualinit_fin_step 1e-4 --primalinit_init_step 1e-1 --primalinit_fin_step 1e-3 --parent_init"

    if use_autostrat:
        cut_specs += " --hard_overhead {}"
        spfw_specs += " --hard_overhead {}"

    cut_iters_overhead = [(100, 6), (600, 25), (1650, 100)]
    spfw_iters_overhead = [(1000, 25), (4000, 100)]

    command_list = []
    for pdprops, nn, batch_size, solver_batch in zip(pdprops_list, nn_names, batch_size_list, max_solver_batch_list):
        for it, overhead in cut_iters_overhead:
            algo_specs = cut_specs.format(it, overhead)
            command_list.append(
                cifar_oval_runner_command(gpu_id, cpus, timeout, pdprops, nn, algo_specs, batch_size, solver_batch))
        for it, overhead in spfw_iters_overhead:
            algo_specs = spfw_specs.format(it, overhead)
            command_list.append(
                cifar_oval_runner_command(gpu_id, cpus, timeout, pdprops, nn, algo_specs, batch_size, solver_batch))

    for idx, ccommand in enumerate(command_list):
        if n_splits > 1 and idx % n_splits != modulo_do:
            continue
        print(ccommand)
        os.system(ccommand)


if __name__ == "__main__":

    # Example: python scripts/anderson/run_anderson_bab_cifar.py --gpu_id 0 --cpus 0-5 --experiment iclr --modulo 2 --modulo_do 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--modulo', type=int, help='Numbers of jobs to split the dataset over (e.g., one per GPU).',
                        default=1)
    parser.add_argument('--modulo_do', type=int, help='Which job_id is this one.', default=0)
    parser.add_argument('--gpu_id', type=str, help='Argument of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--cpus', type=str, help='Argument of taskset -c')
    args = parser.parse_args()

    iclr_experiments(args.gpu_id, args.cpus, n_splits=args.modulo, modulo_do=args.modulo_do, use_autostrat=True)
    jmlr_experiments(args.gpu_id, args.cpus, n_splits=args.modulo, modulo_do=args.modulo_do, use_autostrat=True)
