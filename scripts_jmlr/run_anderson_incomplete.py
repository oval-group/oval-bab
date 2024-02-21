import os
import argparse


def run_incomplete_verification(gpu, cpus, experiment):

    if experiment == "cifar_sgd":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/anderson_cifar_bound_comparison.py " \
                  "./models/cifar_sgd_8px.pth 0.01960784313 ~/results/sgd8 --from_intermediate_bounds"
    elif experiment == 'cifar_madry':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    "python scripts_jmlr/anderson_cifar_bound_comparison.py " \
                    "./models/cifar_madry_8px.pth 0.04705882352 ~/results/madry8 " \
                  "--from_intermediate_bounds"
    elif experiment == 'mnist_kw':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/anderson_mnist_bound_comparison.py " \
                  "0.15 ~/results/mnist_wide --from_intermediate_bounds"
    elif experiment == 'eran_convsmall':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/eran_convsmall_incomplete_verification_experiment.py " \
                  "~/results/eran_convsmall --algorithm bab_prox --n_steps 5;" + \
                  f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/eran_convsmall_incomplete_verification_experiment.py " \
                  "~/results/eran_convsmall --algorithm bab_as --n_steps 5" + \
                  f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/eran_convsmall_incomplete_verification_experiment.py " \
                  "~/results/eran_convsmall --algorithm bab_spfw --n_steps 5"
    elif experiment == 'fc_7000':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  "python scripts_jmlr/anderson_fc_cifar_bound_comparison.py " \
                  "~/results/anderson_jmlr_fc --from_intermediate_bounds"

    print(command)
    os.system(command)

if __name__ == "__main__":

    # Example: python scripts/run_anderson_incomplete.py --gpu_id 0 --cpus 0-3 --experiment cifar_sgd

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, help='Argument of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--cpus', type=str, help='Argument of taskset -c, 4 cpus are required for Gurobi')
    parser.add_argument('--experiment', type=str, help='Which experiment to run',
                        choices=["cifar_sgd", "cifar_madry", "mnist_kw", "all", "eran_convsmall", "fc_7000"])
    args = parser.parse_args()

    if args.experiment != "all":
        run_incomplete_verification(args.gpu_id, args.cpus, args.experiment)
    else:
        for experiment in ["cifar_sgd", "cifar_madry", "mnist_kw", "all", "eran_convsmall", "fc_7000"]:
            run_incomplete_verification(args.gpu_id, args.cpus, experiment)

