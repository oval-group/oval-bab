import argparse
import os
import torch
import time
import json
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.propagation import Propagation
from plnn.explp_solver.solver import ExpLP
from bounding_utils import make_elided_models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tools.bab_tools.vnnlib_utils as vnnlib_utils
import pandas as pd
from tools.bab_tools.model_utils import one_vs_all_from_model
from tools.bab_tools.bab_runner import bab_from_json


def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--algorithm', type=str, help='Which algorithm to run', default='spfw', choices=[
        'spfw', 'as', 'prox', 'bab_as', 'bab_prox', 'bab_spfw'])
    parser.add_argument('--n_steps', type=int, help='N of steps of the algorithm to run.')
    args = parser.parse_args()

    network_filename = './models/onnx/convSmallRELU__PGDK.onnx'
    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(network_filename)
    assert model_correctness
    supported = vnnlib_utils.is_supported_model(model)
    assert supported
    model = torch.nn.Sequential(*[lay.cuda() for lay in model.children()])

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    elided_models = make_elided_models(model, True)

    test = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, pin_memory=True)

    n_images = int(1e3)
    record_name = args.target_directory + f'/eran_convsmall_{args.algorithm}_{args.n_steps}.pkl'
    if os.path.isfile(record_name):
        graph_df = pd.read_pickle(record_name)
    else:
        indices = list(range(n_images))
        graph_df = pd.DataFrame(index=indices, columns=["Idx", "verified", "time"])
        graph_df.to_pickle(record_name)

    for idx, (X, y) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue
        if idx >= n_images:
            # only first 1k images are considered
            break

        graph_df = pd.read_pickle(record_name)
        if pd.isna(graph_df.loc[idx]['Idx']) == False:
            print(f'the {idx}th element is done')
            continue

        elided_model = elided_models[y.item()]

        eps = 2./255.
        domain = torch.stack([X.squeeze(0) - eps,
                              X.squeeze(0) + eps], dim=-1).unsqueeze(0).cuda()

        elided_model = torch.nn.Sequential(*[lay.cuda() for lay in elided_model.children()])
        intermediate_net = Propagation([lay for lay in elided_model], type="best_prop",
                                       params={"best_among": ["KW", "crown"]},
                                       max_batch=int(1e5))

        ## Proximal methods
        if args.algorithm == "prox":
            optprox_params = {
                'nb_total_steps': args.n_steps,
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
            optprox_net = SaddleLP([lay for lay in elided_model])
            optprox_start = time.time()
            with torch.no_grad():
                optprox_net.set_decomposition('pairs', 'KW')
                optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
                intermediate_net.define_linear_approximation(
                    domain, no_conv=False, override_numerical_errors=True)
                optprox_net.build_model_using_bounds(
                    domain, (intermediate_net.lower_bounds, intermediate_net.upper_bounds))
                ub = optprox_net.compute_lower_bound(upper_bound=True, full_batch_asymmetric=True)
            optprox_end = time.time()
            optprox_time = optprox_end - optprox_start
            optprox_ubs = ub.cpu()

            graph_df.loc[idx]["Idx"] = idx
            graph_df.loc[idx]["verified"] = (optprox_ubs.max() <= 0).item()
            graph_df.loc[idx]["time"] = optprox_time


        ## SP-FW
        elif args.algorithm == "spfw":
            step_size_dict = {
                'type': 'fw',
                'fw_start': 10
            }
            explp_params = {
                "anderson_algorithm": "saddle",
                "nb_iter": args.n_steps,
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
            exp_net = ExpLP(
                [lay for lay in elided_model], params=explp_params, use_preactivation=True, fixed_M=True)
            exp_start = time.time()
            with torch.no_grad():
                intermediate_net.define_linear_approximation(
                    domain, no_conv=False, override_numerical_errors=True)
                exp_net.build_model_using_bounds(
                    domain, (intermediate_net.lower_bounds, intermediate_net.upper_bounds))
                ub = exp_net.compute_lower_bound(upper_bound=True, full_batch_asymmetric=True)
            exp_end = time.time()
            exp_time = exp_end - exp_start
            exp_ubs = ub.cpu()

            graph_df.loc[idx]["Idx"] = idx
            graph_df.loc[idx]["verified"] = (exp_ubs.max() <= 0).item()
            graph_df.loc[idx]["time"] = exp_time

        elif args.algorithm == 'as':
            explp_params = {
                "nb_iter": args.n_steps,
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
            exp_net = ExpLP(
                [lay for lay in elided_model], params=explp_params, use_preactivation=True, fixed_M=True)
            exp_start = time.time()
            with torch.no_grad():
                intermediate_net.define_linear_approximation(
                    domain, no_conv=False, override_numerical_errors=True)
                exp_net.build_model_using_bounds(
                    domain, (intermediate_net.lower_bounds, intermediate_net.upper_bounds))
                ub = exp_net.compute_lower_bound(upper_bound=True, full_batch_asymmetric=True)
            exp_end = time.time()
            exp_time = exp_end - exp_start
            exp_ubs = ub.cpu()

            graph_df.loc[idx]["Idx"] = idx
            graph_df.loc[idx]["verified"] = (exp_ubs.max() <= 0).item()
            graph_df.loc[idx]["time"] = exp_time

        elif "bab" in args.algorithm:

            verif_layers = one_vs_all_from_model(
                torch.nn.Sequential(*[lay.cpu() for lay in model.children()]), y, domain=domain.squeeze(0),
                max_solver_batch=10000, use_ib=True, gpu=True)

            timeout = 300  # time out BaB after 300s

            return_dict = dict()
            config = args.algorithm.split("_")[1]
            with open(f'scripts_jmlr/{config}.json') as json_file:
                json_params = json.load(json_file)

            bab_start = time.time()
            bab_from_json(json_params, verif_layers, domain.squeeze(0), return_dict, None, instance_timeout=timeout,
                          gpu=True, max_batches=args.n_steps, return_bounds_if_timeout=True)
            bab_end = time.time()
            bab_time = bab_end - bab_start
            del json_params

            graph_df.loc[idx]["Idx"] = idx
            graph_df.loc[idx]["verified"] = (return_dict['min_lb'] >= 0).item()
            graph_df.loc[idx]["time"] = bab_time

        graph_df.to_pickle(record_name)

    print(f"Pandas tables with results are available at {record_name}")

if __name__ == '__main__':
    main()
