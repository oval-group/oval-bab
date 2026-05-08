import torch
import copy

import plnn.branch_and_bound.utils as bab
from plnn.proxlp_solver import utils
import time
from math import floor, ceil
from typing import List, Optional

class BranchingDecision:
    """
    Represents a single decision made during the branching process in a branch-and-bound algorithm.

    Attributes:
        node (layer, relu) [int, int]: The node at which the decision was made.
        choice (int): The choice made at the node. It is a binary choice (0 [blocking] or 1 [passing]).

    Methods:
        __str__(): Returns a string representation of the BranchingDecision instance.
    """
    def __init__(self, node, choice):
        self.node = node
        self.choice = choice

    def __str__(self):
        return f"Node: {self.node}, Choice: {self.choice}"


class BranchingHistory:
    """
    A class to maintain a history of branching decisions in a branch-and-bound algorithm.
    There is one class for each ReLUDomain object, and it stores the sequence
    of branching decisions made to reach the current domain.

    Attributes:
    -----------
    _decision_list : List[BranchingDecision]
        A private list that stores the sequence of branching decisions made.

    """
    def __init__(self):
        self._decision_list: List[BranchingDecision] = []

    def update(self, new_decision: BranchingDecision):
        self._decision_list.append(new_decision)

    def __str__(self):
        decisions = ""
        for decision in self._decision_list:
            decisions += f"{decision}\n"
        return decisions

class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision assigned to ReLUs. These are reflected in
    the lower and upper bounds for each pre-activation variable (lower_all, upper_all).
    A non-negative lower bound on a given pre-activation implies the ReLU is passing, a non-positive upper bound that
    it is blocking.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.
    """
    def __init__(self, lb, ub, lb_all, up_all, parent_solution=None,
                 parent_ub_point=None, parent_depth=0, c_imp=0, c_imp_avg=0, dec_thr=0, hard_info=None, domain=None,
                 branching_history: Optional[BranchingHistory] = None):
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.parent_solution = parent_solution
        self.parent_ub_point = parent_ub_point
        self.depth = parent_depth + 1
        self.domain = domain

        # keep running improvement average
        avg_coeff = 0.5
        self.impr_avg = (1 - avg_coeff) * c_imp_avg + avg_coeff * c_imp if c_imp_avg != 0 else c_imp
        # Estimate number of children with the current LB net ("easy") and the next ("hard")
        if hard_info is not None:
            self.easy_away, self.hard_away = bab.estimate_branches(
                lb.item(), self.impr_avg, dec_thr, hard_info)
        self.branching_history = branching_history

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.lower_bound = self.lower_bound.cpu()
        self.upper_bound = self.upper_bound.cpu()
        self.lower_all = [lbs.cpu() for lbs in self.lower_all]
        self.upper_all = [ubs.cpu() for ubs in self.upper_all]
        if self.parent_solution is not None:
            self.parent_solution.to_cpu()
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.cpu()
        if self.domain is not None:
            self.domain = self.domain.cpu()
        return self

    def to_device(self, device):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        self.lower_all = [lbs.to(device) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device) for ubs in self.upper_all]
        if self.parent_solution is not None:
            self.parent_solution.to_device(device)
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.to(device)
        if self.domain is not None:
            self.domain = self.domain.to(device)
        return self


def relu_bab(intermediate_dict, out_bounds_dict, brancher, domain, decision_bound, eps=1e-4, early_terminate=False,
             ub_method=None, timeout=float("inf"), gurobi_dict=None, max_cpu_subdomains=None, start_time=None,
             return_bounds_if_timeout=False, max_batches=None, return_ibs_root=False, precomputed_ibs=None,
             leaves_branching_history_list = None, store_history=False):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network. Splits according to KW.
    Does ReLU activation splitting (not domain splitting, the domain will remain the same throughout)

    Assumes that the last layer is a single neuron.

    `intermediate_dict`: dictionary containing classes pertaining to intermediate bound computations,
                         see tools/bounds_from_onnx.py and json files in bab_configs/
    `out_bounds_dict`: dictionary containing classes pertaining to output bound computations,
                         see tools/bounds_from_onnx.py and json files in bab_configs/
    `brancher`: instance of BranchingChoice used to compute the branching decisions
    `ub_method'      : Class to run adversarial attacks to create an UB
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `domain`: the interval domain of the BaB problem (lower and upper bounds provided as entries in the last dim)
    `decision_bound`: If not None, stop the search if the UB and LB are both superior or both inferior to this value.
    `batch_size`: The number of domain lower/upper bounds computations done in parallel at once (on a GPU) is
                    batch_size*2
    `gurobi_dict`: dictionary containing gurobi-related information: whether () gurobi needs to be used (executes on  cpu)
                 - "gurobi_out_lbs", whether gurobi needs to be used (beyond potential MIP calls, see "mip_net")
                 - "p", number of cpus over which gurobi is executed in parallel (spawns threads)
                 - "mip_net", None or an AndersonLinearizedNetwork used as a MIP. Useful to compute tight global UBs
                              when few ambiguous neurons remain. None disables this.
                 - "mip_n_ambiguous", number of remaining ambiguous neurons triggering the MIP call
    `early_terminate`: use heuristic for early termination in case of almost certain timeout
    `max_cpu_subdomains`: how many subdomains we can store in cpu memory
    `timeout`: number of seconds for which BaB can run
    `start_time`: timestamp for when to start counting timeout for (if None, computed internally)
    `return_bounds_if_timeout`: whether to return the LB/UB on the global minimum in case of timeout
    `max_batches`: whether to only run BaB only for at most max_batches batch iterations
    `return_ibs_root`: whether to return the IBs computed for the root node (useful in case BaB is called multiple
                       times on the same network)
    `precomputed_ibs`: provides a tuple of lists with (lower_bounds, upper_bounds) for pre-computed intermediate bounds
                       at the root (useful in case BaB is called multiple times on the same network)
    `leaves_branching_history_list`: list of BranchingHistory objects, each containing the branching history of a domain
    `store_history`: whether to store the branching history
    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    if max_cpu_subdomains != 'inf':
        # How many domains (in the domains list) we can keep in CPU memory.
        max_domains_cpu = int(2e5) if max_cpu_subdomains is None else max_cpu_subdomains
        # How many domains per file (if the total number of domains exceeds max_domain_cpu)
        dumped_domain_blocksize = int(max_domains_cpu/2)
    else:
        max_domains_cpu = None
        dumped_domain_blocksize = None
    n_stratifications = len(out_bounds_dict["nets"])
    stratified_bab = n_stratifications > 1  # whether to use two separate nets for last layer bounds
    if decision_bound is None:
        early_terminate = False

    # Retrieve information concerning the first (and second, if available) last layer bounding network.
    current_net = 0
    bounds_net, net_info, next_net_info, use_auto_iters, batch_size = bab.get_lb_net_info(
        out_bounds_dict, current_net, n_stratifications)

    start_time = time.time() if start_time is None else start_time

    if gurobi_dict:
        p = gurobi_dict["p"]
        gurobi = gurobi_dict["gurobi_out_lbs"]
        if "mip_net" not in gurobi_dict:
            gurobi_dict["mip_net"] = None
        if "mip_n_ambiguous" not in gurobi_dict:
            gurobi_dict["mip_n_ambiguous"] = 0
    else:
        p = 1
        gurobi = False
        gurobi_dict = {"gurobi_out_lbs": gurobi, "p": p, "mip_net": None, "mip_n_ambiguous": 0}
    if gurobi and p > 1:
        send_nets = []
        for cnet in out_bounds_dict["nets"]:
            send_nets.append(cnet["net"])
        if gurobi_dict["mip_net"] is not None:
            send_nets.append(gurobi_dict["mip_net"])
        send_nets = tuple(send_nets)
        cpu_servers, server_queue, instruction_queue, barrier = bab.spawn_cpu_servers(p, send_nets)
        gurobi_dict.update({'server_queue': server_queue, 'instruction_queue': instruction_queue,
                            'barrier': barrier, 'cpu_servers': cpu_servers})
    else:
        gurobi_dict.update({'server_queue': None, 'instruction_queue': None, 'barrier': None, 'cpu_servers': None})

    # do initial computation for the network as it is (batch of size 1: there is only one domain)
    # get intermediate bounds
    if "use_lb" in intermediate_dict and intermediate_dict["use_lb"]:
        intermediate_dict["loose_ib"] = {"net": bounds_net}
    intermediate_net = intermediate_dict["loose_ib"]["net"]
    intermediate_net.define_linear_approximation(domain.unsqueeze(0), override_numerical_errors=True)

    assert intermediate_net.lower_bounds[-1].shape[-1] == 1, "Expecting network to have a single scalar output."

    intermediate_lbs = copy.deepcopy(intermediate_net.lower_bounds)
    intermediate_ubs = copy.deepcopy(intermediate_net.upper_bounds)
    testnode_dict = None

    if decision_bound is not None and (intermediate_lbs[-1] > decision_bound or intermediate_ubs[-1] < decision_bound):
        bab.join_children(gurobi_dict, timeout)
        print(f"Global LB: {intermediate_lbs[-1]}; Global UB: {intermediate_ubs[-1]}")
        # any point is yielding a negative UB, if an UB on the max problem is positive (if no counterexample, all 0s)
        ub_point = utils.apply_transforms(
            intermediate_net.input_transforms,
            (intermediate_net.input_domain.select(-1, 0) + intermediate_net.input_domain.select(-1, 1)) / 2,
            inverse=True)
        # leaves_branching_history_list is returned as None since we are at the root of the BaB (early return)
        out_history = leaves_branching_history_list if store_history else None
        return intermediate_lbs[-1], intermediate_ubs[-1], ub_point, nb_visited_states, out_history

    if precomputed_ibs is None:
        if "tight_ib" in intermediate_dict and intermediate_dict["tight_ib"] is not None and \
                intermediate_dict["tight_ib"]["net"] is not None:

            # Compute tighter intermediate bounds for ambiguous neurons.
            tighten_ambiguous_ibs(
                domain.unsqueeze(0), intermediate_lbs, intermediate_ubs, intermediate_dict["tight_ib"]["net"])

            if decision_bound is not None and \
                    (intermediate_lbs[-1] > decision_bound or intermediate_ubs[-1] < decision_bound):
                bab.join_children(gurobi_dict, timeout)
                print(f"Global LB: {intermediate_lbs[-1]}; Global UB: {intermediate_ubs[-1]}")
                # leaves_branching_history_list is returned as None since we are at the root of the BaB (early return)
                out_history = leaves_branching_history_list if store_history else None
                return intermediate_lbs[-1], intermediate_ubs[-1], \
                    torch.zeros_like(intermediate_net.lower_bounds[0]), nb_visited_states, out_history
    else:
        # if tight pre-computed IBs are passed, no need to re-compute them
        passed_lbs, passed_ubs = precomputed_ibs
        for x_idx in range(min(len(passed_lbs), len(intermediate_lbs) - 1)):
            # retain best bounds and update intermediate bounds from batch
            intermediate_lbs[x_idx] = torch.max(passed_lbs[x_idx], intermediate_lbs[x_idx])
            intermediate_ubs[x_idx] = torch.min(passed_ubs[x_idx], intermediate_ubs[x_idx])

    primal_feasibility = bab.check_primal_infeasibility(intermediate_lbs, intermediate_ubs)
    if not primal_feasibility.all():
        print("********************************************* Numerically unstable property, skip")
        # leaves_branching_history_list is returned as None since we are at the root of the BaB (early return)
        out_history = leaves_branching_history_list if store_history else None
        return None, None, None, nb_visited_states, out_history

    if return_ibs_root:
        # leaves_branching_history_list is returned as None since we are at the root of the BaB (early return)
        out_history = leaves_branching_history_list if store_history else None
        return intermediate_lbs, intermediate_ubs, None, None, out_history

    print('computing last layer bounds')
    # compute last layer bounds with a more expensive network
    if not gurobi:
        bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))
    else:
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab.subproblems_to_cpu(
            domain.unsqueeze(0), intermediate_lbs, intermediate_ubs, squeeze=True)
        bounds_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))

    if intermediate_dict["joint_ib"]:
        # Jointly optimize over IBs while doing last layer lower bounding.
        global_lb, updated_ilbs, updated_iubs = bounds_net.compute_lower_bound(
            node=(-1, 0), counterexample_verification=True)
        for x_idx in range(len(intermediate_lbs)-1):
            intermediate_lbs[x_idx] = torch.max(updated_ilbs[x_idx], intermediate_lbs[x_idx])
            intermediate_ubs[x_idx] = torch.min(updated_iubs[x_idx], intermediate_ubs[x_idx])
        global_ub = intermediate_ubs[-1]
    else:
        global_lb, global_ub = bounds_net.compute_lower_bound(counterexample_verification=True)

    expected_improvement = torch.zeros_like(global_lb)
    # Stratified bounding related.
    if stratified_bab:
        # Dummy place-holder to avoid switching to harder bounding algorithms before we actually estimate their gain.
        next_net_info["lb_impr"] = -1 * torch.ones_like(global_lb).cpu()

    intermediate_lbs[-1] = global_lb
    intermediate_ubs[-1] = global_ub
    intermediate_net_device = domain.device

    # retrieve bounds info from the bounds network
    global_ub_point = bounds_net.get_lower_bound_network_input()
    global_ub = bounds_net.net(global_ub_point)
    if (not out_bounds_dict["do_ubs"]) and (not intermediate_dict["joint_ib"]):
        bounds_net.children_init = bounds_net.children_init.get_lb_init_only()

    print(f"Global LB: {global_lb}; Global UB: {global_ub}")
    print('decision bound', decision_bound)
    if decision_bound is not None and (global_lb > decision_bound or global_ub < decision_bound):
        bab.join_children(gurobi_dict, timeout)
        out_history = leaves_branching_history_list if store_history else None
        return global_lb, global_ub, global_ub_point, nb_visited_states, out_history

    if gurobi_dict["mip_net"] is not None and \
            bab.get_n_ambiguous(intermediate_lbs, intermediate_ubs).item() <= gurobi_dict["mip_n_ambiguous"]:
        # run a MILP solver at the root and terminate
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab.subproblems_to_cpu(
            domain.unsqueeze(0), intermediate_lbs, intermediate_ubs, squeeze=True)
        gurobi_dict["mip_net"].build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
        _, global_lb, bab_nb_states = gurobi_dict["mip_net"].solve_mip(timeout=timeout, insert_cuts=False)
        ub_input = gurobi_dict["mip_net"].get_lower_bound_network_input()
        out_history = leaves_branching_history_list if store_history else None
        return global_lb, gurobi_dict["mip_net"].net(ub_input), ub_input, bab_nb_states, out_history

    start_by_root = not (store_history and leaves_branching_history_list is not None and len(leaves_branching_history_list) > 0)
    if not start_by_root:
        # If leaves_branching_history_list is provided, domains are initialised to contain a candidate domain per
        #  entry of leaves_branching_history_list. These are initialised from domain, intermediate_lbs and
        #  intermediate_ubs, with each of them having one branching decision from leaves_branching_history_list
        #  applied to it.
        leaves_domains = []  # the leaves from a previous BaB call need to be bounded before being branched upon

        # Dict of sets storing for each layer index, the batch entries that are splitting a ReLU there
        leaves_branching_layer_log = {}
        for idx in range(-1, len(intermediate_lbs) - 1):
            leaves_branching_layer_log[idx] = set()

        domain_stack, lbs_stacks, ubs_stacks = bab.create_subproblem_stacks(
            len(leaves_branching_history_list), domain.unsqueeze(0).cpu(),
            [lb.cpu() for lb in intermediate_lbs], [ub.cpu() for ub in intermediate_ubs],
            intermediate_net_device)

        for c_idx, branching_history in enumerate(leaves_branching_history_list):
            # clone info relative to the root before applying branching decisions from previous leaves
            c_dom_lb_all = [lb[c_idx].unsqueeze(0) for lb in lbs_stacks]
            c_dom_ub_all = [ub[c_idx].unsqueeze(0) for ub in ubs_stacks]
            c_domain = domain_stack[c_idx].unsqueeze(0)

            for branching_decision in branching_history._decision_list:
                node = branching_decision.node
                choice = branching_decision.choice
                leaves_branching_layer_log[node[0]] |= {c_idx, }
                brancher.split_subdomain(node, choice, 0, c_domain, c_dom_lb_all, c_dom_ub_all)

            candidate_domain = ReLUDomain(global_lb, global_ub, c_dom_lb_all, c_dom_ub_all,
                                          parent_solution=copy.deepcopy(bounds_net.children_init),
                                          dec_thr=(
                                              decision_bound if (decision_bound is not None) else global_ub.cpu()),
                                          hard_info=next_net_info, domain=c_domain,
                                          branching_history=branching_history).to_cpu()

            leaves_domains.append(candidate_domain)
        domains = []
    if start_by_root:
        candidate_domain = ReLUDomain(global_lb, global_ub, intermediate_lbs, intermediate_ubs,
                                      parent_solution=bounds_net.children_init,
                                      dec_thr=(decision_bound if (decision_bound is not None) else global_ub.cpu()),
                                      hard_info=next_net_info, domain=domain.unsqueeze(0),
                                      branching_history=BranchingHistory() if store_history else None).to_cpu()
        domains = [candidate_domain]
        leaves_domains = []
        leaves_branching_layer_log = {}
    picked_leaves = 0

    # once the provided leaves_branching_history_list has been used, it is re-initialised to null, as there
    #  are currently no leaves.
    leaves_branching_history_list = [] if store_history else None

    dumped_domain_filelblist = []  # store filenames and lbs for blocks of domains that are stored on disk
    harder_domains = []
    mip_domains = []
    using_mip = False
    next_net_buffer = False

    bound_time, branch_time, n_iter, infeasible_count, expans_factor = 0, 0, 0, 0, 2
    while global_ub - global_lb > eps:
        print(f"New batch at {time.time() - start_time}[s]")
        n_iter += 1
        bounds_net_device = global_lb.device
        # Check if we have run out of time.
        # Timeout a bit earlier than necessary to account for the overhead of exiting the scope.
        if (time.time() - start_time) + 1.10 * (bound_time + branch_time) > timeout or \
                (max_batches is not None and n_iter > max_batches):
            bab.join_children(gurobi_dict, timeout)
            bab.delete_dumped_domains(dumped_domain_filelblist)

            # if there are any domains left, all the remaining domains are leaves (timed-out call)
            if store_history:
                if len(dumped_domain_filelblist) > 0:
                    # NOTE: when there are domains stored on disk, the leaves are not returned. The assumption is that
                    # it is undesirable to store leaves when there are so many that you would need to store to disk.
                    leaves_branching_history_list = None
                else:
                    if len(domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in domains])
                    if len(harder_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in harder_domains])
                    if len(mip_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in mip_domains])
                    # Add subproblems from leaves_domains that haven't been processed yet
                    if len(leaves_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in leaves_domains])

            if return_bounds_if_timeout:
                return global_lb, global_ub, global_ub_point, nb_visited_states, \
                    leaves_branching_history_list
            else:
                return None, None, None, nb_visited_states, leaves_branching_history_list

        bounding_leaves = bool(leaves_domains)
        domains_str = f"Number of domains {len(domains)}"
        domains_str += f" Number of harder domains {len(harder_domains)}" if stratified_bab else ""
        domains_str += f" Number of previous leaves domains {len(leaves_domains)}" if bounding_leaves else ""
        domains_str += f" Number of MIP domains {len(mip_domains)}" if gurobi_dict["mip_net"] else ""
        print(domains_str)
        if max_cpu_subdomains != 'inf' and len(dumped_domain_filelblist) > 0:
            print(f"Number of pickled domains {len(dumped_domain_filelblist) * dumped_domain_blocksize}")

        effective_batch_size = min(batch_size, len(domains) if not bounding_leaves else len(leaves_domains))
        print(f"effective_batch_size {effective_batch_size}")
        # flag indicating whether these batch of subproblems will be split
        do_branching = (not using_mip) and (not bounding_leaves)

        stack_example = bab.get_subproblem_stacks_entry(
            0, domain.unsqueeze(0), bounds_net.lower_bounds, bounds_net.upper_bounds)
        domain_stack, lbs_stacks, ubs_stacks = bab.create_subproblem_stacks(
            effective_batch_size, *stack_example, intermediate_net_device)
        batch_branching_history_list = []
        depth_list = []
        parent_lb_list = []
        impr_avg_list = []

        if next_net_buffer:
            # Ignore initialization if in the transitioning buffer.
            bounds_net.children_init = bab.ParentInit()
        bounds_net.children_init.to_device(intermediate_net_device)
        stack_size = effective_batch_size * 2 if do_branching else effective_batch_size
        parent_init_stacks = bounds_net.children_init.as_stack(stack_size)

        # Pick a batch of domains from the subproblem queue.
        for batch_idx in range(effective_batch_size):
            # Pick a domain to branch over and remove that from our current list of
            # domains. Also, potentially perform some pruning on the way.
            candidate_domain = bab.pick_out(domains if not bounding_leaves else leaves_domains,
                                            global_ub.cpu()).to_device(intermediate_net_device)

            # Populate the batch of subproblems for batched branching
            bab.set_subproblem_stacks_entry(batch_idx, domain_stack, lbs_stacks, ubs_stacks, candidate_domain.domain,
                                            candidate_domain.lower_all, candidate_domain.upper_all)

            # collect branching related information
            if do_branching:
                # binary branching, hence duplicating entries (one per child)
                if stratified_bab:
                    parent_lb_list.extend([candidate_domain.lower_bound, candidate_domain.lower_bound])
                    impr_avg_list.extend([candidate_domain.impr_avg, candidate_domain.impr_avg])
                depth_list.extend([candidate_domain.depth, candidate_domain.depth])
                if store_history:
                    batch_branching_history_list.extend([candidate_domain.branching_history,
                                                         copy.deepcopy(candidate_domain.branching_history)])
            else:
                # no branching, no need to duplicate
                if stratified_bab:
                    parent_lb_list.append(candidate_domain.lower_bound)
                    impr_avg_list.append(candidate_domain.impr_avg)
                depth_list.append(candidate_domain.depth)
                if store_history:
                    batch_branching_history_list.append(candidate_domain.branching_history)

            # Testnode selection for the quantities related to automatically inferring the number of iters.
            if current_net == 0:
                get_testnode = (use_auto_iters and testnode_dict is None and candidate_domain.depth >= 4)
            else:
                get_testnode = (use_auto_iters and testnode_dict is None and (not next_net_buffer))
            if get_testnode:
                # Use as test node the first picked node with a depth of 4 (this way split constraints are represented)
                # If stratifying, it's applied only to the tighter bounding algorithm
                testnode_dict = {"domain": candidate_domain.domain, "intermediate_lbs": candidate_domain.lower_all,
                                 "intermediate_ubs": candidate_domain.upper_all}
                if out_bounds_dict["parent_init"]:
                    testnode_dict["pinit"] = candidate_domain.parent_solution

                # Bounding improvement estimations for automatic number of bounds iterations, and for stratified bounds
                # Automatic number of iterations.
                tighter_lb = bab.assess_impr_margin(
                    bounds_net, candidate_domain.domain, candidate_domain.lower_all, candidate_domain.upper_all,
                    net_info, intermediate_dict, c_lb=candidate_domain.lower_bound)
                net_info["+iters_lb"] = tighter_lb

            # Stratified bounding related.
            # Postpone evaluation to when domains might be indeed marked as hard.
            evaluate_auto_strat = next_net_info and next_net_info["lb_impr"] == -1 and \
                                  ((expans_factor**(candidate_domain.easy_away+1))/batch_size >
                                   next_net_info["hard_overhead"])
            # if doing autoiters, start autostrat only at the max number of iterations for the looser bounds
            auto_iters_check = net_info["max_iters_reached"] or (not use_auto_iters)
            if evaluate_auto_strat and auto_iters_check and not using_mip:
                # Automatically infer stratification parameters by estimating the hard bounding gain.
                if not gurobi:
                    bounds_net.initialize_from(copy.deepcopy(candidate_domain.parent_solution))
                    bounds_net.build_model_using_bounds(
                        candidate_domain.domain, (candidate_domain.lower_all, candidate_domain.upper_all))
                    next_net_info["net"].initialize_from(copy.deepcopy(candidate_domain.parent_solution))
                    next_net_info["net"].build_model_using_bounds(
                        candidate_domain.domain, (candidate_domain.lower_all, candidate_domain.upper_all))
                else:
                    cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab.subproblems_to_cpu(
                        candidate_domain.domain, candidate_domain.lower_all, candidate_domain.upper_all,
                        squeeze=True)
                    next_net_info["net"].build_model_using_bounds(
                        cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
                    bounds_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
                if use_auto_iters:
                    # If using autoiters, need to recompute a LB for the test problem (the tightness might have changed)
                    looser_lb = bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
                else:
                    looser_lb = candidate_domain.lower_bound
                tighter_lb = next_net_info["net"].compute_lower_bound(node=(-1, 0), counterexample_verification=True)
                lb_impr = (tighter_lb - looser_lb).mean()
                print(f"Hard bounding improvement over looser LB: {lb_impr}")
                next_net_info["lb_impr"] = lb_impr.cpu()

            # get parent's dual solution from the candidate domain
            parent_init_stacks.set_stack_parent_entries(
                candidate_domain.parent_solution, batch_idx, do_branching=do_branching)
        bounds_net.unbuild()

        # Dict of sets storing for each layer index, the batch entries that are splitting a ReLU there
        branching_layer_log = {}
        for idx in range(-1, len(lbs_stacks) - 1):
            branching_layer_log[idx] = set()

        if do_branching:
            # BRANCHING: compute branching choices
            branch_start = time.time()
            branching_decision_list = brancher.branch(
                domain_stack, lbs_stacks, ubs_stacks, parent_net=bounds_net, parent_init=parent_init_stacks)
            branch_time = time.time() - branch_start
            print(f"Branching requires {branch_time}[s]")

            # DOMAIN SPLITTING.
            # Create stacks for the bounding of the split subproblems (duplicates the subdomains, with the two copies for
            # the i-th subdomain in position 2*i, 2*i + 1).
            domain_stack, lbs_stacks, ubs_stacks = bab.create_split_stacks(
                domain_stack, lbs_stacks, ubs_stacks)

            for batch_idx, branching_decision in enumerate(branching_decision_list):
                branching_layer_log[branching_decision[0]] |= {2*batch_idx, 2*batch_idx+1}

                # Binary branching.
                for choice in [0, 1]:
                    nb_visited_states += 1

                    # split the domain with the current branching decision
                    brancher.split_subdomain(
                        branching_decision, choice, 2*batch_idx + choice, domain_stack, lbs_stacks, ubs_stacks)

                    # Append the current branching decision and choice to the branching history of the corresponding domain
                    if store_history:
                        batch_branching_history_list[2*batch_idx + choice].update(BranchingDecision(branching_decision, choice))

        elif bounding_leaves:
            # retrieve the branching layer log (where to compute intermediate bounds from) from what computed before
            # the BaB loop
            c_filter = lambda x: (x >= picked_leaves) & (x < (picked_leaves + effective_batch_size))
            remainder = lambda x: (x % picked_leaves) if picked_leaves > 0 else x
            for key in leaves_branching_layer_log:
                branching_layer_log[key] = {remainder(a) for a in leaves_branching_layer_log[key] if c_filter(a)}
            picked_leaves += effective_batch_size

        print(f"Running Nb states visited: {nb_visited_states}")
        print(f"N. infeasible nodes {infeasible_count}")

        relu_start = time.time()
        # compute the bounds on the batch of splits, at once
        dom_ub, dom_lb, dom_ub_point, dom_lb_all, dom_ub_all, dual_solutions, expected_improvement = \
            compute_bounds(intermediate_dict, bounds_net, branching_layer_log, domain_stack, lbs_stacks,
            ubs_stacks, parent_init_stacks, out_bounds_dict["parent_init"], gurobi_dict, expected_improvement,
            testnode_dict=testnode_dict, compute_last_ubs=out_bounds_dict["do_ubs"], net_info=net_info,
            using_mip=using_mip, timeout=timeout, start_time=start_time, dec_threshold=decision_bound)

        # update the global upper bound (if necessary) comparing to the best of the batch
        batch_ub, batch_ub_point_idx = torch.min(dom_ub, dim=0)
        batch_ub_point = dom_ub_point[batch_ub_point_idx]
        if batch_ub < global_ub:
            global_ub = batch_ub
            global_ub_point = batch_ub_point

        # sequentially add all the domains to the queue (ordered list)
        batch_global_lb = torch.min(dom_lb, dim=0)[0]
        added_domains = 0
        for batch_idx in range(dom_lb.shape[0]):

            if dom_lb[batch_idx] == float('inf') or dom_ub[batch_idx] == float('inf'):
                infeasible_count += 1

                # Infeasible problems for the previous BaB call won't necessarily be infeasible for the next.
                #  As such, they are marked as leaves.
                if store_history:
                    leaves_branching_history_list.append(batch_branching_history_list[batch_idx])

            elif dom_lb[batch_idx] < min(global_ub, (decision_bound if (decision_bound is not None) else global_ub)):
                added_domains += 1
                c_dom_lb_all = [lb[batch_idx].unsqueeze(0) for lb in dom_lb_all]
                c_dom_ub_all = [ub[batch_idx].unsqueeze(0) for ub in dom_ub_all]

                c_dual_solutions = dual_solutions.get_stack_entry(batch_idx)
                if stratified_bab:
                    # store statistics that determine when to move to tighter last layer bounding
                    dom_to_add = ReLUDomain(
                        dom_lb[batch_idx].unsqueeze(0), dom_ub[batch_idx].unsqueeze(0), c_dom_lb_all,
                        c_dom_ub_all, parent_solution=c_dual_solutions,
                        parent_depth=depth_list[batch_idx], c_imp_avg=impr_avg_list[batch_idx],
                        c_imp=dom_lb[batch_idx].item() - parent_lb_list[batch_idx].item(),
                        dec_thr=(decision_bound if (decision_bound is not None) else global_ub.cpu()),
                        hard_info=next_net_info, domain=domain_stack[batch_idx].unsqueeze(0),
                        branching_history=batch_branching_history_list[batch_idx] if store_history else None
                    ).to_cpu()
                else:
                    dom_to_add = ReLUDomain(
                        dom_lb[batch_idx].unsqueeze(0), dom_ub[batch_idx].unsqueeze(0),
                        c_dom_lb_all, c_dom_ub_all, parent_solution=c_dual_solutions,
                        parent_depth=depth_list[batch_idx], domain=domain_stack[batch_idx].unsqueeze(0),
                        branching_history=batch_branching_history_list[batch_idx] if store_history else None).to_cpu()

                if gurobi_dict["mip_net"] is not None and not using_mip and \
                        bab.get_n_ambiguous(c_dom_lb_all, c_dom_ub_all).item() <= gurobi_dict["mip_n_ambiguous"]:
                    # if a MIP network is provided, use it to verify subdomains with ambiguous neurons below threshold
                    bab.add_domain(dom_to_add, mip_domains)
                # if the problem is hard, add "difficult" domains to the hard queue
                elif next_net_info and bab.is_difficult_domain(dom_to_add, next_net_info, expansion=expans_factor):
                    bab.add_domain(dom_to_add, harder_domains)
                else:
                    if next_net_buffer:
                        # use a buffer so that when the buffer is empty, we can use the hard problem's parent init
                        bab.add_domain(dom_to_add, harder_domains)
                    else:
                        bab.add_domain(dom_to_add, domains)

            else:
                # the pruned domain is a leaf: add to leaves_branching_history_list
                if store_history:
                    leaves_branching_history_list.append(batch_branching_history_list[batch_idx])

        divider = 2 if do_branching else 1
        expans_factor = max(float(added_domains) / (dom_lb.shape[0] / divider), 1.0)
        print(f"Batch expansion factor: {expans_factor}")
        bound_time = time.time() - relu_start
        print('A batch of relu splits requires: ', bound_time)
        if next_net_info and "postpone_batches" in next_net_info:
            next_net_info["postpone_batches"] -= 1

        # Remove domains clearly on the right side of the decision threshold: our goal is to which side of it is the
        # minimum, no need to know more for these domains.
        prune_value = min(global_ub.cpu() - eps,
                          (decision_bound if (decision_bound is not None) else global_ub.cpu()) + eps)
        domains, pruned_domains = bab.prune_domains(domains, prune_value)
        if store_history:
            for pd in pruned_domains:
                leaves_branching_history_list.append(pd.branching_history)

        if max_cpu_subdomains != 'inf':
            # read/write domains from secondary memory if necessary.
            domains = bab.dump_domains_to_file(domains, dumped_domain_filelblist, max_domains_cpu, dumped_domain_blocksize)
            dumped_domain_filelblist = bab.get_domains_from_file(domains, dumped_domain_filelblist, dumped_domain_blocksize)

        if len(leaves_domains) == 0:
            # Update global LB when there are no parent leaves left to handle.

            if len(domains) + len(harder_domains) + len(mip_domains) > 0:

                # find the lowest LB across the three domains
                dom_candidate = domains[0] if domains else None
                harder_candidate = harder_domains[0] if harder_domains else None
                mip_candidate = mip_domains[0] if mip_domains else None
                lb_candidate = bab.min_ignoring_none(bab.min_ignoring_none(dom_candidate, harder_candidate),
                                                     mip_candidate)
                global_lb = lb_candidate.lower_bound.to(bounds_net_device)

                # and compare it with the min across any pickled domains
                if dumped_domain_filelblist:
                    dumped_lb_min = min(dumped_domain_filelblist, key=lambda x: x[1])[1].to(bounds_net_device)
                    global_lb = min(global_lb, dumped_lb_min)
            else:
                # If we've run out of domains, it means we included no newly splitted domain
                if batch_global_lb <= global_ub:
                    global_lb = batch_global_lb
                elif decision_bound is not None:
                    global_lb = torch.ones_like(global_lb) * (
                        prune_value.to(global_lb.device) if torch.is_tensor(prune_value) else prune_value)

        if harder_domains or next_net_buffer:
            harder_domains, pruned_domains = bab.prune_domains(harder_domains, prune_value)
            if store_history:
                # pruned domains are leaves and hence stored
                for pd in pruned_domains:
                    leaves_branching_history_list.append(pd.branching_history)

        if mip_domains:
            mip_domains, pruned_domains = bab.prune_domains(mip_domains, prune_value)
            if store_history:
                # pruned domains are leaves and hence stored
                for pd in pruned_domains:
                    leaves_branching_history_list.append(pd.branching_history)

        # Switch to the harder domains if there's a next net or we're in the transition buffer
        if len(domains) == 0 and (next_net_info or next_net_buffer) and len(harder_domains) > 0:
            domains = harder_domains
            harder_domains = []
            # Check whether it's worth switching network
            if next_net_buffer:
                # The buffer has been emptied -- will now initialize from parent (handled automatically via PInit)
                print('tighter bounding buffer emptied')
                next_net_buffer = False
            else:
                do_switch, expected_batches = bab.switch_to_hard(domains, next_net_info, batch_size)
                if do_switch:
                    print('shifting to tighter bounding')

                    # Move to the following net in the provided list, updating the corresponding dictionaries of info
                    next_net_info["net"].lower_bounds = bounds_net.lower_bounds
                    next_net_info["net"].upper_bounds = bounds_net.upper_bounds
                    current_net += 1
                    bounds_net, net_info, next_net_info, use_auto_iters, batch_size = bab.get_lb_net_info(
                        out_bounds_dict, current_net, n_stratifications)

                    if out_bounds_dict["parent_init"]:
                        # use a buffer so that when the buffer is empty, we can use the hard problem's parent init
                        next_net_buffer = True
                    if gurobi:
                        bab.gurobi_switch_bounding_net(gurobi_dict, current_net)
                    if use_auto_iters:
                        testnode_dict = None
                else:
                    # Postpone adding to the harder queue for expected_batches batches.
                    next_net_info["postpone_batches"] = expected_batches
        elif len(domains) == 0 and (gurobi_dict["mip_net"] is not None) and (not using_mip) and mip_domains:
            # switch to a MIP for solving the remaining domains
            print('shifting to a MIP solver for the remaining domains')
            domains = mip_domains
            mip_domains = []

            if p > 1:
                if not gurobi:
                    # if using multiple threads, spawn them
                    send_nets = (gurobi_dict["mip_net"],)
                    cpu_servers, server_queue, instruction_queue, barrier = bab.spawn_cpu_servers(p, send_nets)
                    gurobi_dict.update({'server_queue': server_queue, 'instruction_queue': instruction_queue,
                                        'barrier': barrier, 'cpu_servers': cpu_servers})
                else:
                    # instruct the other running threads to use the MIP net
                    bab.gurobi_switch_bounding_net(gurobi_dict, -1)

            using_mip = True
            gurobi = True
            gurobi_dict["gurobi_out_lbs"] = True
            # move global tensors onto CPU for consistency
            global_ub = global_ub.cpu()
            global_lb = global_lb.cpu()
            expected_improvement = expected_improvement.cpu()

            gurobi_dict["mip_net"].lower_bounds = bounds_net.lower_bounds
            gurobi_dict["mip_net"].upper_bounds = bounds_net.upper_bounds

            mip_net_info = {
                "nets": [{
                    "net": gurobi_dict["mip_net"],
                    "batch_size": batch_size,
                    "auto_iters": False
                }],
                'do_ubs': False,
                'parent_init': False
            }
            out_bounds_dict = mip_net_info
            bounds_net, net_info, next_net_info, use_auto_iters, batch_size = bab.get_lb_net_info(mip_net_info, 0, 1)

        # If early_terminate is True, we return early if we predict that the property won't be verified within the time
        # (if the estimated time to cross the decision threshold + to deplete the bounds goes over the timeout)
        t_to_timeout = timeout - (time.time() - start_time)
        early_terminate_lhs = ((decision_bound if (decision_bound is not None) else global_ub)
                               - global_lb) / expected_improvement + len(domains) / batch_size
        if max_cpu_subdomains != 'inf':
            # consider pickled domains
            early_terminate_lhs += len(dumped_domain_filelblist) * dumped_domain_blocksize / batch_size
        if next_net_info is not None:
            # add time to deplete harder domains as well
            early_terminate_lhs += len(harder_domains) * next_net_info["hard_overhead"] / next_net_info[
                "batch_size"]
        if early_terminate and n_iter > 5 and (early_terminate_lhs > t_to_timeout / bound_time):
            if store_history:
                # all the remaining domains when timing out are leaves
                if len(dumped_domain_filelblist) > 0:
                    # NOTE: when there are domains stored on disk, the leaves are not returned. The assumption is that
                    # it is undesirable to store leaves when there are so many that you would need to store to disk.
                    leaves_branching_history_list = None
                else:
                    if len(domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in domains])
                    if len(harder_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in harder_domains])
                    if len(mip_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in mip_domains])
                    # Add subproblems from leaves_domains that haven't been processed yet
                    if len(leaves_domains) > 0:
                        leaves_branching_history_list.extend([dom.branching_history for dom in leaves_domains])

            print(
                f'early timeout with expected improvement: {expected_improvement} with {t_to_timeout} [s] remaining.')
            bab.join_children(gurobi_dict, timeout)
            bab.delete_dumped_domains(dumped_domain_filelblist)

            if return_bounds_if_timeout:
                return global_lb, global_ub, global_ub_point, nb_visited_states, \
                    leaves_branching_history_list
            else:
                return None, None, None, nb_visited_states, leaves_branching_history_list

        # run attacks
        # Try falsification only in the first 50 batch iterations.
        if (ub_method is not None and ((decision_bound is None) or (global_ub > decision_bound))) and n_iter < 50:
            global_ub, global_ub_point = run_attack(ub_method, dom_ub, dom_ub_point, global_ub, global_ub_point)

        if decision_bound is None:
            domains, pruned_domains= bab.prune_domains(domains, global_ub.cpu() - eps)
            if store_history:
                # pruned domains are leaves and hence stored
                for pd in pruned_domains:
                    leaves_branching_history_list.append(pd.branching_history)

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")
        # Stopping criterion
        stop_value = (decision_bound if (decision_bound is not None) else (global_ub - eps))
        if global_lb >= stop_value:
            break
        elif global_ub < stop_value:
            break

    bab.join_children(gurobi_dict, timeout)

    print(f"Terminated in {time.time() - start_time}[s]; {nb_visited_states} nodes.")
    print(f"Infeasible count: {infeasible_count}")
    print(f"N. batches: {n_iter}")

    bab.delete_dumped_domains(dumped_domain_filelblist)

    if store_history:
        # all the remaining domains when terminating are leaves
        if len(dumped_domain_filelblist) > 0:
            # NOTE: when there are domains stored on disk, the leaves are not returned. The assumption is that
            # it is undesirable to store leaves when there are so many that you would need to store to disk.
            leaves_branching_history_list = None
        else:
            if len(domains) > 0:
                leaves_branching_history_list.extend([dom.branching_history for dom in domains])
            if len(harder_domains) > 0:
                leaves_branching_history_list.extend([dom.branching_history for dom in harder_domains])
            if len(mip_domains) > 0:
                leaves_branching_history_list.extend([dom.branching_history for dom in mip_domains])
            # Add subproblems from leaves_domains that haven't been processed yet
            if len(leaves_domains) > 0:
                leaves_branching_history_list.extend([dom.branching_history for dom in leaves_domains])

    return global_lb, global_ub, global_ub_point, nb_visited_states, leaves_branching_history_list


def compute_bounds(intermediate_dict, bounds_net, branching_layer_log, splitted_domain, splitted_lbs,
                   splitted_ubs, parent_init_stacks, parent_init_flag, gurobi_dict, expected_improvement,
                   testnode_dict=None, compute_last_ubs=False, net_info=None, using_mip=False, timeout=None,
                   start_time=None, dec_threshold=None):
    """
    Split domain according to branching decision and compute all the necessary quantities for it.
    Splitting on the input domain will never happen as it'd be done on l1-u1, rather than l0-u0 (representing the
    conditioned input domain). So conditioning is not problematic, here.
    :param intermediate_dict: Dictionary of networks (and info on how to select them) used for intermediate bounds
    :param bounds_net: Network used for last bounds
    :param branching_layer_log: List of sets storing for each layer index, the set of batch entries that are
        splitting a ReLU there (stored like x_idx-1)
    :param choice: 0/1 for whether to clip on a blocking/passing ReLU
    :param splitted_lbs: list of tensors for the (pre-activation) lower bounds relative to all the activations of the
    network, for all the domain batches
    :param splitted_ubs:list of tensors for the (pre-activation) upper bounds relative to all the activations of the
        network, for all the domain batches
    :param parent_init_stacks:list of tensors to use as dual variable initialization in the last layer solver
    :return: domain UB, domain LB, net input point that yielded UB, updated old_lbs, updated old_ubs
    :param parent_init_flag: whether to initialize the bounds optimisation from the parent node
    :param gurobi_dict: dictionary containing information for gurobi's (possibly parallel) execution
    :param compute_last_ubs: whether to compute UBs for the last layer (not on its min, actual UBs).
    :param expected_improvement: running avg of the expected improvement over IB last bounds (used for bounding budget).
    :param net_info: information concerning bounds_net (useful to automatically infer the no. of iters)
    :param testnode_dict: dict containing the info for the BaB node used to assess tightness (used only for auto_iters)
    :param using_mip: whether we're currently using the MIP solver on the subdomains
    :param timeout: global timeout in seconds (useful to time out inner MIP calls)
    :param start_time: start time (useful to time out inner MIP calls)
    :param dec_threshold: decision threshold (useful to early-stop inner MIP calls)
    """
    # whether to keep the intermediate bounding fixed throughout BaB (after root)
    if not intermediate_dict["fixed_ib"]:
        # update intermediate bounds after the splitting
        splitted_lbs, splitted_ubs = compute_intermediate_bounds(
            intermediate_dict, branching_layer_log, splitted_domain, splitted_lbs, splitted_ubs, parent_init_stacks)

    # get the new last-layer bounds after the splitting
    if not gurobi_dict["gurobi_out_lbs"]:

        # Increase/decrease the number of iterations for the last layer bounding based on expected_improvement.
        if net_info["auto_iters"] and testnode_dict is not None:
            bab.do_auto_iters(bounds_net, expected_improvement, net_info, testnode_dict, intermediate_dict)

        if parent_init_flag:
            bounds_net.initialize_from(parent_init_stacks)
        # compute all last layer bounds in parallel
        bounds_net.build_model_using_bounds(splitted_domain, (splitted_lbs, splitted_ubs))

        if not compute_last_ubs:
            # here, not computing upper bounds to save memory and time
            if intermediate_dict["joint_ib"]:
                # Jointly optimize over IBs while doing last layer lower bounding.
                updated_lbs, updated_ilbs, updated_iubs = bounds_net.compute_lower_bound(
                    node=(-1, 0), counterexample_verification=True)
                for x_idx in range(len(splitted_lbs)-1):
                    splitted_lbs[x_idx] = torch.max(updated_ilbs[x_idx], splitted_lbs[x_idx])
                    splitted_ubs[x_idx] = torch.min(updated_iubs[x_idx], splitted_ubs[x_idx])
            else:
                updated_lbs = bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        else:
            # can catch more infeasible domains by computing UBs as well (not recommended)
            updated_lbs, _ = bounds_net.compute_lower_bound(node=(-1, None), counterexample_verification=True)

        #  Update running average of the expected improvement over output LB for this split, considering only the
        #  candidate domain with the lowest LB (post-update) in the batch
        worst_impr, layer_ind = torch.min(updated_lbs.masked_fill(torch.isnan(updated_lbs), float("inf")), 0)
        current_improvement = (updated_lbs[layer_ind] - splitted_lbs[-1][layer_ind]).mean()

        expected_improvement = current_improvement * 0.5 + expected_improvement * 0.5 if expected_improvement > 0 else \
            current_improvement

        splitted_lbs[-1] = torch.max(updated_lbs, splitted_lbs[-1])
        # evaluate the network at the lower bound point
        dom_ub_point = bounds_net.get_lower_bound_network_input()
        dual_solutions = bounds_net.children_init
    else:
        # compute them one by one
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = compute_last_bounds_cpu(
            bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict, using_mip=using_mip, timeout=timeout,
            start_time=start_time, dec_threshold=dec_threshold)

    # retrieve bounds info from the bounds network: the lower bounds are the output of the bound calculation, the upper
    # bounds are computed by evaluating the network at the lower bound points.
    dom_lb_all = splitted_lbs
    dom_ub_all = splitted_ubs
    dom_lb = splitted_lbs[-1]

    # TODO: do we need any alternative upper bounding strategy for the dual algorithms?
    dom_ub = bounds_net.net(dom_ub_point)

    # check that all over-approximation-based upper bounds are larger than the corresponding lower bound.
    # If not, infeasible domain, and return +inf to have the bound pruned.
    primal_feasibility = bab.check_primal_infeasibility(dom_lb_all, dom_ub_all)
    dom_lb = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_lb), dom_lb)
    dom_ub = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_ub), dom_ub)

    return dom_ub, dom_lb, dom_ub_point, dom_lb_all, dom_ub_all, dual_solutions, expected_improvement


def compute_intermediate_bounds(intermediate_dict, branching_layer_log, splitted_domain, intermediate_lbs,
                                intermediate_ubs, parent_init_stacks):
    # compute intermediate bounds for the current batch, leaving out unnecessary computations
    # (those before the splitted relus)

    # get minimum layer idx where branching is happening
    intermediate_net = intermediate_dict["loose_ib"]["net"]
    min_branching_layer = len(intermediate_lbs)-1
    for branch_lay_idx in branching_layer_log.keys():
        if branching_layer_log[branch_lay_idx]:
            min_branching_layer = branch_lay_idx
            break

    # Dict of sets storing for each layer index (-1 is input), the batch entries splitting a ReLU there or onwards
    cumulative_branching_layer_log = {}
    for count, branch_lay_idx in enumerate(branching_layer_log.keys()):
        cumulative_branching_layer_log[branch_lay_idx] = branching_layer_log[branch_lay_idx]
        if count > 0:
            cumulative_branching_layer_log[branch_lay_idx] |= cumulative_branching_layer_log[branch_lay_idx-1]

    loose_layer_range = list(range(min_branching_layer+2, len(intermediate_lbs)-1))
    # Update intermediate bounds only for ambiguous ReLUs.
    neurons_to_opt = {}
    batch_size = intermediate_lbs[0].shape[0]
    for lay_idx in loose_layer_range:
        ambiguous = ((intermediate_lbs[lay_idx] < 0) & (intermediate_ubs[lay_idx] > 0)).view(batch_size, -1)
        max_ambiguous = ambiguous.sum(1).max().item()  # Max. no. of ambiguous activations across batches
        neurons_to_opt[lay_idx] = torch.topk(ambiguous.type(intermediate_lbs[0].dtype), max_ambiguous)[1]
    intermediate_bounds_subroutine(intermediate_net, splitted_domain, intermediate_lbs, intermediate_ubs,
                                   loose_layer_range, cumulative_branching_layer_log, neurons_to_opt=neurons_to_opt)

    # Tighten selected intermediate bounds with a more expensive network.
    if intermediate_dict["tight_ib"] is not None and intermediate_dict["tight_ib"]["net"] is not None:
        # Optimise neurons of layer before last
        before_last = len(intermediate_lbs)-2
        tigth_layer_range = list(range(before_last, len(intermediate_lbs)-1))

        _, indices_to_opt = torch.topk(parent_init_stacks.get_bounding_score(before_last), intermediate_dict["k"])
        neurons_to_opt = {before_last: indices_to_opt}
        intermediate_bounds_subroutine(intermediate_dict["tight_ib"]["net"], splitted_domain, intermediate_lbs,
                                       intermediate_ubs, tigth_layer_range, cumulative_branching_layer_log,
                                       neurons_to_opt=neurons_to_opt)
        intermediate_dict["tight_ib"]["net"].unbuild()

    return intermediate_lbs, intermediate_ubs


def intermediate_bounds_subroutine(bounding_net, splitted_domain, intermediate_lbs, intermediate_ubs, layer_range,
                                   cumulative_branching_layer_log, neurons_to_opt=None):
    # intermediate bound computation subroutine: compute bounds for the layers in layer_range using bounding_net,
    # but only for the subset of subproblems described by cumulative_branching_layer_log.
    # Moreover a non-None neurons_to_opt (dictionary indexed by x_idx) restricts the indices of which intermediate
    # bounds to compute for each subproblem (each entry is a 2D tensor --conv layers are vectorised-- or list of lists).

    for x_idx in layer_range:
        if not cumulative_branching_layer_log[x_idx-2]:
            continue
        active_batch_ids = list(cumulative_branching_layer_log[x_idx-2])
        if neurons_to_opt is not None and neurons_to_opt[x_idx].shape[1] == 0:
            continue
        sub_batch_intermediate_lbs = [lbs[active_batch_ids] for lbs in intermediate_lbs]
        sub_batch_intermediate_ubs = [ubs[active_batch_ids] for ubs in intermediate_ubs]
        bounding_net.internal_init()

        bounding_net.build_model_using_bounds(
            splitted_domain[active_batch_ids],
            (sub_batch_intermediate_lbs, sub_batch_intermediate_ubs), build_limit=x_idx)
        updated_lbs, updated_ubs = bounding_net.compute_lower_bound(
            node=(x_idx, None if neurons_to_opt is None else neurons_to_opt[x_idx][active_batch_ids]),
            counterexample_verification=True, override_numerical_errors=True)

        # retain best bounds and update intermediate bounds from batch
        intermediate_lbs[x_idx][active_batch_ids] = torch.max(updated_lbs, intermediate_lbs[x_idx][active_batch_ids])
        intermediate_ubs[x_idx][active_batch_ids] = torch.min(updated_ubs, intermediate_ubs[x_idx][active_batch_ids])
        intermediate_lbs[x_idx][active_batch_ids], intermediate_ubs[x_idx][active_batch_ids] = \
            bab.override_numerical_bound_errors(intermediate_lbs[x_idx][active_batch_ids],
                                                intermediate_ubs[x_idx][active_batch_ids])
        bounding_net.unbuild()


def compute_last_bounds_cpu(bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict,
                            using_mip=False, timeout=None, start_time=None, dec_threshold=None):
    # Compute the last layer bounds on (multiple, if p>1) cpu over the batch domains (used for Gurobi).

    # Retrieve execution specs.
    p = gurobi_dict["p"]
    server_queue = gurobi_dict["server_queue"]
    instruction_queue = gurobi_dict["instruction_queue"]
    barrier = gurobi_dict["barrier"]

    if p == 1:
        batch_indices = list(range(splitted_lbs[0].shape[0]))
        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs)
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, batch_indices, using_mip=using_mip,
            timeout=timeout, start_time=start_time, dec_threshold=dec_threshold)
    else:
        # Full synchronization after every batch.
        barrier.wait()

        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs, share=True)

        busy_processors = 0
        total_batch_size = cpu_splitted_lbs[0].shape[0]
        for worker_n in range(1, p):
            is_busy, slice_indices = bab.cpu_subproblem_allocator(worker_n, total_batch_size, p)
            if is_busy:
                # Send bounding jobs to the busy cpu servers.
                instruction_queue.put((cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices,
                                       using_mip, timeout, start_time, dec_threshold))
                busy_processors += 1
            else:
                # Keep the others idle.
                instruction_queue.put(("idle",))

        # Execute the last sub-batch of bounds on this cpu core.
        _, slice_indices = bab.cpu_subproblem_allocator(0, total_batch_size, p)
        splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices, share=True,
            using_mip=using_mip, timeout=timeout, start_time=start_time, dec_threshold=dec_threshold)

        # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
        dom_ub_point = c_dom_ub_point[0].unsqueeze(0).repeat(((total_batch_size,) + (1,) * (c_dom_ub_point.dim() - 1)))
        dual_solutions = c_dual_solutions.as_stack(total_batch_size)
        dom_ub_point[slice_indices] = c_dom_ub_point
        dual_solutions.set_stack_parent_entries(c_dual_solutions, slice_indices, do_branching=False)

        for _ in range(busy_processors):
            # Collect bounding jobs from cpu servers.
            splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions, slice_indices = \
                server_queue.get(True)

            # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
            dom_ub_point[slice_indices] = c_dom_ub_point
            dual_solutions.set_stack_parent_entries(c_dual_solutions, slice_indices, do_branching=False)

    return splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions


def run_attack(ub_method, dom_ub, dom_ub_point, global_ub, global_ub_point):
    # Use the 10 best points from the LB initialization amongst the falsification initializers
    val, ind = torch.topk(dom_ub, dim=0, k=min(10, dom_ub.size()[0]))
    init_tensor = dom_ub_point[ind.squeeze()]

    # perform attacks for property falsification
    adv_examples, is_adv, scores = ub_method.create_adv_examples(
        return_criterion='one', gpu=True, multi_targets=True, init_tensor=init_tensor)

    # Update upper bound and its associated point.
    attack_ub, attack_point_idx = torch.min(scores, dim=0)
    attack_point = adv_examples[attack_point_idx]
    if is_adv.sum() > 0:
        print("Found a counter-example.")

    if attack_ub.to(global_ub.device) < global_ub:
        global_ub = attack_ub.to(global_ub.device)
        global_ub_point = attack_point.to(global_ub.device)
    return global_ub, global_ub_point


def tighten_ambiguous_ibs(domain, intermediate_lbs, intermediate_ubs, ibs_net):
    # Compute tighter intermediate bounds for ambiguous neurons.
    dummy_branching_log = {idx: [0] for idx in range(-1, len(intermediate_lbs) - 1)}
    neurons_to_opt = {}
    loose_layer_range = list(range(2, len(intermediate_lbs) - 1))
    for lay_idx in loose_layer_range:
        ambiguous = ((intermediate_lbs[lay_idx] < 0) & (intermediate_ubs[lay_idx] > 0)).view(domain.shape[0], -1)
        max_ambiguous = ambiguous.sum(1).max().item()  # Max. no. of ambiguous activations across batches
        neurons_to_opt[lay_idx] = torch.topk(ambiguous.type(intermediate_lbs[0].dtype), max_ambiguous)[1]
    intermediate_bounds_subroutine(
        ibs_net, domain, intermediate_lbs, intermediate_ubs, loose_layer_range, dummy_branching_log,
        neurons_to_opt=neurons_to_opt)