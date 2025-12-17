import torch
from torch import nn
from torch.nn import functional as F
from tools.custom_torch_modules import shape_transforms, build_unified_math_transforms, \
    parse_post_linear_math_transform, supported_transforms, Mul
from plnn.proxlp_solver.utils import override_numerical_bound_errors, apply_transforms


class NaiveNetwork:
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def remove_maxpools(self, domain):
        from plnn.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers

    def get_lower_bound(self, domain):
        self.do_interval_analysis(domain)
        return self.lower_bounds[-1]

    def _find_first_linear(self, domain):
        # Find first layer.
        first_layer = 0
        pre_linear_transform = None
        for idx, clayer in enumerate(self.layers):
            if isinstance(clayer, (nn.Linear, nn.Conv2d)):
                first_layer = idx
                break
            elif isinstance(clayer, supported_transforms):
                if isinstance(clayer, Mul):
                    # negative multiplications would need a more careful handling of the input bounds
                    assert (clayer.const >= 0).all()
                if pre_linear_transform is None:
                    pre_linear_transform = [clayer]
                else:
                    pre_linear_transform.append(clayer)

        # Setup the bounds on the inputs, in the transformed space.
        self.input_transforms = pre_linear_transform
        self.input_domain = torch.stack(
            [apply_transforms(pre_linear_transform, domain.select(-1, 0), no_fold=True),
             apply_transforms(pre_linear_transform, domain.select(-1, 1), no_fold=True)], dim=-1)
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        return first_layer, l_0, u_0

    def do_interval_analysis(self, inp_domain, override_numerical_errors=False, cdebug=False):

        first_layer, l_0, u_0 = self._find_first_linear(inp_domain)

        self.lower_bounds = [-torch.ones_like(l_0)]
        self.upper_bounds = [torch.ones_like(u_0)]

        if cdebug:
            inp_ex = (torch.zeros_like(l_0).uniform_() * (u_0 - l_0) + l_0)
            x = inp_ex

        current_lb = l_0
        current_ub = u_0
        to_skip = 0
        start_idx = first_layer
        for cidx, layer in enumerate(self.layers[start_idx:]):
            lay_idx = cidx + start_idx
            if to_skip > 0:
                to_skip -= 1
                continue
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                # check if math operations trail this layer
                post_linear_math_transform, to_skip = build_unified_math_transforms(self.layers[lay_idx + 1:], to_skip)
                if post_linear_math_transform is not None:
                    layer = parse_post_linear_math_transform(layer, *post_linear_math_transform)

                abs_weights = torch.abs(layer.weight)
                if type(layer) is nn.Linear:
                    center = (current_ub + current_lb) @ layer.weight.t() / 2 + layer.bias
                    offset = (current_ub - current_lb) @ abs_weights.t() / 2
                else:
                    center = F.conv2d((current_ub + current_lb) / 2, layer.weight, layer.bias,
                                      layer.stride, layer.padding, layer.dilation, layer.groups)
                    offset = F.conv2d((current_ub - current_lb) / 2, abs_weights, None,
                                      layer.stride, layer.padding, layer.dilation, layer.groups)

                new_layer_lb = center - offset
                new_layer_ub = center + offset

                if override_numerical_errors:
                    new_layer_lb, new_layer_ub = override_numerical_bound_errors(new_layer_lb, new_layer_ub)
                self.lower_bounds.append(new_layer_lb)
                self.upper_bounds.append(new_layer_ub)
                current_lb = new_layer_lb
                current_ub = new_layer_ub
            elif type(layer) == nn.ReLU:
                current_lb = torch.clamp(current_lb, min=0)
                current_ub = torch.clamp(current_ub, min=0)
            elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d)):
                new_lbs = layer(current_lb)
                new_ubs = layer(current_ub)
                current_lb = new_lbs
                current_ub = new_ubs
                self.lower_bounds.append(new_lbs)
                self.upper_bounds.append(new_ubs)
            elif isinstance(layer, (*shape_transforms, nn.AvgPool2d, nn.ConstantPad2d)):
                # Simply propagate the operations forward
                current_lb = layer(current_lb)
                current_ub = layer(current_ub)
            else:
                raise NotImplementedError
            if cdebug:
                x = layer(x)

        if cdebug:
            self._debug_forward(inp_ex, x)

    def define_linear_approximation(self, inp_domain, override_numerical_errors=False, cdebug=False):
        self.do_interval_analysis(inp_domain, override_numerical_errors=override_numerical_errors, cdebug=cdebug)

    def compute_lower_bound(self, node=(-1, None), upper_bound=False, counterexample_verification=False,
                            override_numerical_errors=False, full_batch_asymmetric=False):
        raise NotImplementedError("compute_lower_bound not implemented for naive implementation of IBP for now,"
                                  "use Propagation with type='naive'")

    def unbuild(self):
        # Release memory by discarding the stored model information.
        self.lower_bounds = []
        self.upper_bounds = []

    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.shape[:-1]
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        sp_shape = (nb_samples, ) + nb_inp
        rand_samples = torch.Tensor(*sp_shape)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.unsqueeze(0).expand(*sp_shape)
        domain_width = domain_width.unsqueeze(0).expand(*sp_shape)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    def _debug_forward(self, inp_ex, x, numerical_tolerance=1e-3):
        assert (self.net(inp_ex) - x).abs().max() < numerical_tolerance

    get_upper_bound = get_upper_bound_random
