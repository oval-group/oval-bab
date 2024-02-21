import argparse
import os
import torch
import time
import copy
from plnn.proxlp_solver.solver import SaddleLP
from plnn.explp_solver.solver import ExpLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from bounding_utils import load_network, make_elided_models, cifar_loaders, dump_bounds
from tools.bab_tools.model_utils import mnist_model, mnist_model_deep

def load_mnist_wide_net(idx, network="wide", mnist_test = None):
    if network == "wide":
        model_name = './models/mnist_wide_kw.pth'
        model = mnist_model()
    else:
        # "deep"
        model_name = './models/mnist_deep_kw.pth'
        model = mnist_model_deep()
    model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transforms.ToTensor())

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()

    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y:
        print('model prediction is incorrect for the given model')
        return None, None, None
    else:
        elided_models = make_elided_models(model, True)
        return x, y, elided_models

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('eps', type=float, help='Epsilon - default: 0.1')
    parser.add_argument('target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    parser.add_argument('--network', type=str,
                        help='which network to use', default="wide", choices=["wide", "deep"])
    args = parser.parse_args()

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    testset_size = int(1e5)
    for idx in range(testset_size):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue
        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)

        X, y, elided_models = load_mnist_wide_net(idx, mnist_test=None)
        if X is None:
            continue
        elided_model = elided_models[y]
        to_ignore = y

        domain = torch.stack([torch.clamp(X.squeeze(0) - args.eps, 0, None),
                              torch.clamp(X.squeeze(0) + args.eps, None, 1.0)], -1).unsqueeze(0)

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"

        # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
        # and optimize only the last layer
        if args.from_intermediate_bounds:
            cuda_elided_model = copy.deepcopy(elided_model).cuda()
            cuda_domain = domain.cuda()
            intermediate_net = SaddleLP([lay for lay in cuda_elided_model])
            with torch.no_grad():
                intermediate_net.set_solution_optimizer('best_inits', ["naive", "KW"])
                intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
                                                             override_numerical_errors=True)
            intermediate_ubs = intermediate_net.upper_bounds
            intermediate_lbs = intermediate_net.lower_bounds

        ## Proximal methods
        for optprox_steps in [400]:
            optprox_params = {
                'nb_total_steps': optprox_steps,
                'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
                'initial_eta': 1e0,
                'final_eta': 5e1,
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True,
                'acceleration_dict': {
                    'momentum': 0.3,  # decent momentum: 0.9 w/ increasing eta
                }
            }
            optprox_target_file = os.path.join(target_dir, f"Proximal_finalmomentum_{optprox_steps}{lin_approx_string}.txt")
            if not os.path.exists(optprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                optprox_net = SaddleLP([lay for lay in cuda_elided_model])
                optprox_start = time.time()
                with torch.no_grad():
                    optprox_net.set_decomposition('pairs', 'KW')
                    optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
                    if not args.from_intermediate_bounds:
                        optprox_net.define_linear_approximation(cuda_domain, no_conv=False)
                        ub = optprox_net.upper_bounds[-1]
                    else:
                        optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = optprox_net.compute_lower_bound()
                optprox_end = time.time()
                optprox_time = optprox_end - optprox_start
                optprox_ubs = ub.cpu()

                del optprox_net
                dump_bounds(optprox_target_file, optprox_time, optprox_ubs)

        ## Gurobi PLANET Bounds
        grb_target_file = os.path.join(target_dir, f"Gurobi{lin_approx_string}-fixed.txt")
        if not os.path.exists(grb_target_file):
            grb_net = LinearizedNetwork([lay for lay in elided_model])
            grb_start = time.time()
            if not args.from_intermediate_bounds:
                grb_net.define_linear_approximation(domain[0], n_threads=4)
                ub = grb_net.upper_bounds[-1]
            else:
                grb_net.build_model_using_bounds(domain[0], ([lbs[0].cpu() for lbs in intermediate_lbs],
                                                          [ubs[0].cpu() for ubs in intermediate_ubs]), n_threads=4)
                _, ub = grb_net.compute_lower_bound(ub_only=True)
            grb_end = time.time()
            grb_time = grb_end - grb_start
            grb_ubs = torch.Tensor(ub).cpu()
            dump_bounds(grb_target_file, grb_time, grb_ubs)

        ## SP-FW
        for spfw_steps in [80, 1000, 2000, 4000, 7500]:
            step_size_dict = {
                'type': 'fw',
                'fw_start': 10
            }
            explp_params = {
                "anderson_algorithm": "saddle",
                "nb_iter": spfw_steps,
                'blockwise': False,
                "step_size_dict": step_size_dict,
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': 1e-1,
                    'final_step_size': 1e-3,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                "primal_init_params": {
                    'nb_bigm_iter': 100,
                    'nb_anderson_iter': 0,
                    'initial_step_size': 1e-1,
                    'final_step_size': 1e-3,
                    'betas': (0.9, 0.999)
                }
            }
            explp_params.update({"bigm": "init", "bigm_algorithm": "adam"})

            spfw_target_file = os.path.join(target_dir, f"SP-FW_{spfw_steps}{lin_approx_string}.txt")
            if not os.path.exists(spfw_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                exp_net = ExpLP(
                    [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True, fixed_M=True)
                exp_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        exp_net.define_linear_approximation(cuda_domain)
                        ub = exp_net.upper_bounds[-1]
                    else:
                        exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = exp_net.compute_lower_bound()
                exp_end = time.time()
                exp_time = exp_end - exp_start
                exp_ubs = ub.cpu()

                del exp_net
                dump_bounds(spfw_target_file, exp_time, exp_ubs)

        ## Cuts
        for cut_steps in [80, 600, 1050, 1650, 2500]:
            explp_params = {
                "nb_iter": cut_steps,
                'bigm': "init",
                'cut': "only",
                "bigm_algorithm": "adam",
                'cut_frequency': 450,
                'max_cuts': 12,
                'cut_add': 2,
                'betas': (0.9, 0.999),
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': 1e-1,
                    'final_step_size': 1e-3,
                    'betas': (0.9, 0.999)
                },
            }
            cut_target_file = os.path.join(target_dir, f"Cuts_{cut_steps}{lin_approx_string}.txt")
            if not os.path.exists(cut_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                exp_net = ExpLP(
                    [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True, fixed_M=True)
                exp_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        exp_net.define_linear_approximation(cuda_domain)
                        ub = exp_net.upper_bounds[-1]
                    else:
                        exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = exp_net.compute_lower_bound()
                exp_end = time.time()
                exp_time = exp_end - exp_start
                exp_ubs = ub.cpu()

                del exp_net
                dump_bounds(cut_target_file, exp_time, exp_ubs)

        ## Cuts + SP-FW
        for spfw_steps, cut_steps in [(5500, 1050)]:
            step_size_dict = {
                'type': 'fw',
                'fw_start': 10
            }
            explp_params = {
                "anderson_algorithm": "saddle",
                "nb_iter": spfw_steps,
                'blockwise': False,
                "step_size_dict": step_size_dict,
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': 1e-1,
                    'final_step_size': 1e-3,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                "primal_init_params": {
                    'nb_bigm_iter': 100,
                    'nb_anderson_iter': 0,
                    'initial_step_size': 1e-1,
                    'final_step_size': 1e-3,
                    'betas': (0.9, 0.999)
                }
            }
            cut_init_params = {
                'cut_frequency': 450,
                'max_cuts': 12,
                'cut_add': 2,
                'nb_iter': cut_steps,
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
            }
            explp_params.update({"cut": "init", "cut_init_params": cut_init_params})

            spfw_target_file = os.path.join(target_dir,
                                            f"SP-FW_Cuts_{spfw_steps}-{cut_steps}{lin_approx_string}.txt")
            if not os.path.exists(spfw_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                exp_net = ExpLP(
                    [lay for lay in cuda_elided_model], params=explp_params, use_preactivation=True,
                    fixed_M=True)
                exp_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        exp_net.define_linear_approximation(cuda_domain)
                        ub = exp_net.upper_bounds[-1]
                    else:
                        exp_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = exp_net.compute_lower_bound()
                exp_end = time.time()
                exp_time = exp_end - exp_start
                exp_ubs = ub.cpu()

                del exp_net
                dump_bounds(spfw_target_file, exp_time, exp_ubs)

        # Big-M supergradient. (iters tuned to take same time as prox)
        for bigm_steps in [850]:
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": bigm_steps,
                'initial_step_size': 1e-1,
                'final_step_size': 1e-3,
                'betas': (0.9, 0.999)
            }
            bigm_target_file = os.path.join(target_dir, f"Big-M_{bigm_steps}{lin_approx_string}.txt")
            if not os.path.exists(bigm_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                bigm_net = ExpLP(
                    [lay for lay in cuda_elided_model], params=bigm_adam_params, use_preactivation=True,
                    fixed_M=True)
                bigm_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        bigm_net.define_linear_approximation(cuda_domain)
                        ub = bigm_net.upper_bounds[-1]
                    else:
                        bigm_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = bigm_net.compute_lower_bound()
                bigm_end = time.time()
                bigm_time = bigm_end - bigm_start
                bigm_ubs = ub.cpu()

                del bigm_net
                dump_bounds(bigm_target_file, bigm_time, bigm_ubs)

        ## Gurobi Anderson Bounds
        for n_cuts in [1]:
            grb_and_target_file = os.path.join(target_dir, f"Anderson-{n_cuts}cuts{lin_approx_string}-fixed.txt")
            if not os.path.exists(grb_and_target_file):
                lp_and_grb_net = AndersonLinearizedNetwork(
                    [lay for lay in elided_model], mode="lp-cut", n_cuts=n_cuts, cuts_per_neuron=True)
                lp_and_grb_start = time.time()
                if not args.from_intermediate_bounds:
                    lp_and_grb_net.define_linear_approximation(domain[0], n_threads=4)
                    ub = lp_and_grb_net.upper_bounds[-1]
                else:
                    lp_and_grb_net.build_model_using_bounds(domain[0], ([lbs[0].cpu() for lbs in intermediate_lbs],
                                                                     [ubs[0].cpu() for ubs in intermediate_ubs]), n_threads=4)
                    _, ub = lp_and_grb_net.compute_lower_bound(ub_only=True)
                lp_and_grb_end = time.time()
                lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
                lp_and_grb_ubs = torch.Tensor(ub).cpu()
                dump_bounds(grb_and_target_file, lp_and_grb_time, lp_and_grb_ubs)


if __name__ == '__main__':
    main()
