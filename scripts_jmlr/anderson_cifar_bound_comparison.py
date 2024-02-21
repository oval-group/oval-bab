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


def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('network_filename', type=str,
                        help='Path to the network')
    parser.add_argument('eps', type=float,
                        help='Epsilon - default: 0.0347. Expressed in the normalized space, so for reporting purtposes '
                             'one should multiply the employed epsilon by 0.225 (the std used in cifar_loaders)')
    parser.add_argument('target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    args = parser.parse_args()
    model = load_network(args.network_filename)

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    elided_models = make_elided_models(model, True)

    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue
        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)
        elided_model = elided_models[y.item()]
        to_ignore = y.item()

        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1).unsqueeze(0)

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
                'initial_eta': 1e1,
                'final_eta': 5e2,
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
        for spfw_steps in [80, 1000, 2000, 4000]:
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
                    'initial_step_size': 1e-2,
                    'final_step_size': 1e-4,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                "primal_init_params": {
                    'nb_bigm_iter': 100,
                    'nb_anderson_iter': 0,
                    'initial_step_size': 1e-2,  # 1e-2,
                    'final_step_size': 1e-5,  # 1e-3,
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

        ## Cuts (iters tuned to take same time as sp-fw + 100 as we use it in complete verification)
        for cut_steps in [80, 100, 600, 1050, 1650]:
            explp_params = {
                "nb_iter": cut_steps,
                'bigm': "init",
                'cut': "only",
                "bigm_algorithm": "adam",
                'cut_frequency': 450,
                'max_cuts': 8,
                'cut_add': 2,
                'betas': (0.9, 0.999),
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': 1e-2,
                    'final_step_size': 1e-4,
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
        for spfw_steps, cut_steps in [(3000, 600)]:
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
                    'initial_step_size': 1e-2,
                    'final_step_size': 1e-4,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                "primal_init_params": {
                    'nb_bigm_iter': 100,
                    'nb_anderson_iter': 0,
                    'initial_step_size': 1e-2,  # 1e-2,
                    'final_step_size': 1e-5,  # 1e-3,
                    'betas': (0.9, 0.999)
                }
            }
            cut_init_params = {
                'cut_frequency': 450,
                'max_cuts': 8,
                'cut_add': 2,
                'nb_iter': cut_steps,
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
            }
            explp_params.update({"cut": "init", "cut_init_params": cut_init_params})

            spfw_target_file = os.path.join(target_dir, f"SP-FW_Cuts_{spfw_steps}-{cut_steps}{lin_approx_string}.txt")
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
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
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

        ## Gurobi Anderson Bounds on neuron subset.
        for n_cuts in [1]:
            for andersonk in [50, 100, 200, 300]:
                grb_and_target_file = os.path.join(
                    target_dir, f"Anderson-{n_cuts}cuts-{andersonk}neurons{lin_approx_string}-fixed.txt")
                if not os.path.exists(grb_and_target_file):
                    lp_and_grb_net = AndersonLinearizedNetwork(
                        [lay for lay in elided_model], mode="lp-cut", n_cuts=n_cuts, cuts_per_neuron=True,
                        neurons_per_layer=andersonk)
                    lp_and_grb_start = time.time()
                    if not args.from_intermediate_bounds:
                        lp_and_grb_net.define_linear_approximation(domain[0], n_threads=4)
                        ub = lp_and_grb_net.upper_bounds[-1]
                    else:
                        lp_and_grb_net.build_model_using_bounds(domain[0],
                                                                ([lbs[0].cpu() for lbs in intermediate_lbs],
                                                                 [ubs[0].cpu() for ubs in intermediate_ubs]),
                                                                n_threads=4)
                        _, ub = lp_and_grb_net.compute_lower_bound(ub_only=True)
                    lp_and_grb_end = time.time()
                    lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
                    lp_and_grb_ubs = torch.Tensor(ub).cpu()
                    dump_bounds(grb_and_target_file, lp_and_grb_time, lp_and_grb_ubs)

        ## Cuts on CPU.
        for cut_steps in [600]:
            explp_params = {
                "nb_iter": cut_steps,
                'bigm': "init",
                'cut': "only",
                "bigm_algorithm": "adam",
                'cut_frequency': 450,
                'max_cuts': 8,
                'cut_add': 2,
                'betas': (0.9, 0.999),
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': 1e-2,
                    'final_step_size': 1e-4,
                    'betas': (0.9, 0.999)
                },
            }
            cut_target_file = os.path.join(target_dir, f"CutsCPU_{cut_steps}{lin_approx_string}.txt")

            if not os.path.exists(cut_target_file):
                exp_net = ExpLP(
                    [lay for lay in copy.deepcopy(elided_model)], params=explp_params, use_preactivation=True,
                    fixed_M=True)
                exp_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        exp_net.define_linear_approximation(domain)
                        ub = exp_net.upper_bounds[-1]
                    else:
                        exp_net.build_model_using_bounds(domain,
                            ([clbs.cpu() for clbs in intermediate_lbs], [cubs.cpu() for cubs in intermediate_ubs]))
                        _, ub = exp_net.compute_lower_bound()
                exp_end = time.time()
                exp_time = exp_end - exp_start

                del exp_net
                dump_bounds(cut_target_file, exp_time, ub)

        ## SP-FW on CPU.
        for spfw_steps in [1000]:
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
                    'initial_step_size': 1e-2,
                    'final_step_size': 1e-4,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                "primal_init_params": {
                    'nb_bigm_iter': 100,
                    'nb_anderson_iter': 0,
                    'initial_step_size': 1e-2,  # 1e-2,
                    'final_step_size': 1e-5,  # 1e-3,
                    'betas': (0.9, 0.999)
                }
            }
            explp_params.update({"bigm": "init", "bigm_algorithm": "adam"})

            spfw_target_file = os.path.join(target_dir, f"SP-FWCPU_{spfw_steps}{lin_approx_string}.txt")
            if not os.path.exists(spfw_target_file):
                exp_net = ExpLP(
                    [lay for lay in copy.deepcopy(elided_model)], params=explp_params, use_preactivation=True,
                    fixed_M=True)
                exp_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        exp_net.define_linear_approximation(domain)
                        ub = exp_net.upper_bounds[-1]
                    else:
                        exp_net.build_model_using_bounds(domain,
                                                         ([clbs.cpu() for clbs in intermediate_lbs],
                                                          [cubs.cpu() for cubs in intermediate_ubs]))
                        _, ub = exp_net.compute_lower_bound()
                exp_end = time.time()
                exp_time = exp_end - exp_start

                del exp_net
                dump_bounds(spfw_target_file, exp_time, ub)

        test_as_sensitivity = True
        if test_as_sensitivity:
            ## Active Set with random cuts.
            for cut_steps in [600, 1050, 1650]:
                explp_params = {
                    "nb_iter": cut_steps,
                    'bigm': "init",
                    'cut': "only",
                    "bigm_algorithm": "adam",
                    'cut_frequency': 450,
                    'random_cuts': True,
                    'max_cuts': 8,
                    'cut_add': 2,
                    'betas': (0.9, 0.999),
                    'initial_step_size': 1e-3,
                    'final_step_size': 1e-6,
                    "init_params": {
                        "nb_outer_iter": 500,
                        'initial_step_size': 1e-2,
                        'final_step_size': 1e-4,
                        'betas': (0.9, 0.999)
                    },
                }
                cut_target_file = os.path.join(target_dir, f"Cuts_randmask_{cut_steps}{lin_approx_string}.txt")
                if not os.path.exists(cut_target_file):
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
                    dump_bounds(cut_target_file, exp_time, exp_ubs)

            ## Active Set with different addition frequencies.
            freq_steps = [(300, [550, 900, 1500]), (600, [650, 1150, 1850])]
            for freq, cut_steps_list in freq_steps:
                for cut_steps in cut_steps_list:
                    explp_params = {
                        "nb_iter": cut_steps,
                        'bigm': "init",
                        'cut': "only",
                        "bigm_algorithm": "adam",
                        'cut_frequency': freq,
                        'random_cuts': False,
                        'max_cuts': 8,
                        'cut_add': 2,
                        'betas': (0.9, 0.999),
                        'initial_step_size': 1e-3,
                        'final_step_size': 1e-6,
                        "init_params": {
                            "nb_outer_iter": 500,
                            'initial_step_size': 1e-2,
                            'final_step_size': 1e-4,
                            'betas': (0.9, 0.999)
                        },
                    }
                    cut_target_file = os.path.join(target_dir, f"Cuts_freq{freq}_{cut_steps}{lin_approx_string}.txt")
                    if not os.path.exists(cut_target_file):
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
                        dump_bounds(cut_target_file, exp_time, exp_ubs)


if __name__ == '__main__':
    main()
