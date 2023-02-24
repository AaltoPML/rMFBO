from __future__ import annotations
import torch
from botorch.models.deterministic import DeterministicModel
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
import math
import argparse
import numpy as np
import itertools
import matplotlib as mpl


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def is_primary_source(x, opposite=False):
    target = 1.0 if not opposite else 0.0
    if x.dim() == 2:
        if x.shape[0]==1:
            return bool(math.isclose(float(x[0,-1]),target))
        else:
            return [bool(math.isclose(float(x[i,-1]),target)) for i in range(x.shape[0])]
    else:
        raise ValueError('Wrong dim tensor')


def parser_bo():

    """
    Parser used to run the algorithm from an already known crn.
    - Output:
        * parser: ArgumentParser object.
    """

    parser = argparse.ArgumentParser(description="Command description.")

    parser.add_argument(
        "-n", "--N_REP", help="int, number of reps for stds", type=int, default=1
    )
    parser.add_argument(
        "-se", "--seed", default=None, help="int, random seed", type=int
    )
    parser.add_argument(
        "-s", "--save", default=None, type=str, help="Save the results on a dict."
    )
    parser.add_argument(
        "-j", "--jointmogp", nargs="*", default=["downsampling"], type=str, help="list of types of MOGP."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose for algorithm",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--B",
        help="BO Budget",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-cv",
        "--cond_var",
        nargs="*",
        type=float,
        default=[0.1],
        help="list of variance threshold values to try.",
    )
    parser.add_argument(
        "-ci",
        "--cond_ig",
        nargs="*",
        type=str,
        default=["0.1"],
        help="list of IG threshold values to try. passing 'adaptive' uses average information gain.",
    )
    parser.add_argument(
        "-co",
        "--costs",
        nargs="*",
        type=float,
        default=[0.2],
        help="list of lower fidelity costs to try.",
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="*",
        type=str,
        default=["rMF-MES"],
        help="list of BO methods to try."
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="*",
        type=str,
        default=["StyblinskiTang"],
        help="list of test functions to optimize."
    )   
    return parser


def build_combinations(N_REP, experiments, costs, methods, conds_var, conds_ig, mogp, seed):

    """Construct the list of combinations to try."""
    combi = []
    for m in methods:
        if "rMF" in m:
            li = [experiments, [m], costs, conds_var, conds_ig, mogp, [(n, seed + n) for n in range(N_REP)]]
            combi.append(list(itertools.product(*li)))
        elif ("rMF" not in m and "MF" in m):
            li = [experiments, [m], costs, [-1], [-1], mogp, [(n, seed + n) for n in range(N_REP)]] # MF methods don't need varying conds.
            combi.append(list(itertools.product(*li)))
        else:
            li = [experiments, [m], [-1], [-1], [-1], mogp, [(n, seed + n) for n in range(N_REP)]] # SF methods don't need varying conds/costs
            combi.append(list(itertools.product(*li)))
    return [c + (i,) for i, c in enumerate(sum(combi, []))] # adding combination number for tracking where parallelization is at currently. ugly but...


def set_matplotlib_params():

    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    # plt.style.use('fivethirtyeight')
    mpl.rc('font', family='serif')
    mpl.rcParams.update({'font.size': 24,
                     'lines.linewidth': 2,
                     'axes.labelsize': 24,  # fontsize for x and y labels
                     'axes.titlesize': 24,
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                     "text.usetex": True,                # use LaTeX to write all text
                     "axes.spines.right": False,
                     "axes.spines.top": False,
                     "axes.spines.left": True,
                     "axes.spines.bottom": True
                     })


def adapt_save_fig(fig, filename="test.pdf"):

    """Remove right and top spines, set bbox_inches and dpi."""

    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


class CostOne(DeterministicModel):
    r"""
    For each (q-batch) element of a candidate set `X`, this module computes a
    cost of the form

        cost = 1

    Example:
        >>> from botorch.models import AffineFidelityCostModel
        >>> from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        >>> cost_model = CostOne(
        >>>    fidelity_weights={6: 1.0}
        >>> )
        >>> cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    """

    def __init__(
        self,
        fidelity_weights= None, # same
        fixed_cost: float = 0.01, # will not be used anyway
    ) -> None:
        r"""
        Args:
            fidelity_weights: A dictionary mapping a subset of columns of `X`
                (the fidelity parameters) to its associated weight in the
                affine cost expression. If omitted, assumes that the last
                column of `X` is the fidelity parameter with a weight of 1.0.
            fixed_cost: The fixed cost of running a single candidate point (i.e.
                an element of a q-batch).
        """
        if fidelity_weights is None:
            fidelity_weights = {-1: 1.0}
        super().__init__()
        self.fidelity_dims = sorted(fidelity_weights)
        self.fixed_cost = fixed_cost
        weights = torch.tensor([fidelity_weights[i] for i in self.fidelity_dims])
        self.register_buffer("weights", weights)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost on a candidate set X.

        Computes a cost of the form

            cost = fixed_cost + sum_j weights[j] * X[fidelity_dims[j]]

        for each element of the q-batch

        Args:
            X: A `batch_shape x q x d'`-dim tensor of candidate points.

        Returns:
            A `batch_shape x q x 1`-dim tensor of costs.
        """

        lin_cost = torch.einsum(
            "...f,f", X[..., self.fidelity_dims], self.weights.to(X)
        )
        lin_cost[:] = 1.
        return self.fixed_cost + lin_cost.unsqueeze(-1)


from problems import * # importing it here instead of before prevents circular imports with main.py and mfbo.py.


def get_problem_settings(experiment: str = "Hartmann", cost: float = .2):

    if experiment=="Hartmann":
        ndim = 6
        problem = Hartmann(noise_std=.01).to(**tkwargs)
        bounds = torch.tensor([[0.0] * (ndim + 1), [1.0] * (ndim + 1)], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost,1.0]
        ninits = [5*ndim,4*ndim]

        # Takeno et al. (2020) setting, budget = 300/5 = 60
        #list_fidelities = [0.2,0.6,1.0]
        #ninits = [6*ndim,3*ndim,2*ndim]

    if experiment=="HartmannCost":
        ndim = 6
        problem = HartmannCost(noise_std=.01)
        bounds = torch.tensor([[0.0] * (ndim + 1), [1.0] * (ndim + 1)], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost,1.0]
        ninits = [5*ndim,4*ndim]

    elif experiment=="HartmannMultiple":
        ndim = 6
        problem = HartmannMultiple(noise_std=0.01)
        bounds = torch.tensor([[0.0] * (ndim + 1), [1.0] * (ndim + 1)], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [0.0, 0.1, 0.8, 1.0]
        ninits = [5*ndim, 5*ndim, 5*ndim, 4*ndim]

    elif experiment=="BraninMultiple":
        ndim = 2
        problem = BraninMultiple(noise_std=0.0001)
        bounds = torch.tensor([[-5.0,0.0,0.0], [10.0,15.0,1.0]], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [0.0, 0.1, 0.8, 1.0]#[0.0, 0.1, 0.8, 1.0]
        ninits = [5*ndim, 5*ndim, 5*ndim, 4*ndim]

    elif experiment=="RosenbrockSinus": #Difficult objective, more budget needed
        ndim = 2
        problem = RosenbrockSinus(dim=ndim, noise_std=0.0001, negate=True)
        bounds = torch.tensor([([-5.0] * ndim)+[0.0], ([5.0] * ndim)+[1.0]], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost, 1.0]
        ninits = [5*ndim, 4*ndim]

    elif experiment=="HartmannRosenbrock":
        ndim = 6
        problem = HartmannRosenbrock(noise_std=.01)
        bounds = torch.tensor([[0.0] * (ndim + 1), [1.0] * (ndim + 1)], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost, 1.0]
        ninits = [5*ndim,4*ndim]
    
    elif experiment=="Branin":
        ndim = 2
        problem = Branin(noise_std=.0001, negate=True).to(**tkwargs)
        bounds = torch.tensor([[-5.0,0.0,0.0], [10.0,15.0,1.0]], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost,1.0]
        ninits = [5*ndim,4*ndim]

    elif experiment=="CurrinNegated":
        ndim = 2
        problem = CurrinNegated(noise_std=0.01)
        bounds = torch.tensor([[0.0,0.0,0.0], [1.0,1.0,1.0]], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost,1.0]
        ninits = [5*ndim,4*ndim]

    elif experiment=="XGB":
        ndim = 5
        problem = XGB(noise_std=None)
        bounds = torch.tensor([[-2.0, -2.0, -1.0, -2.0, -3.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0, 1.0]], **tkwargs)
        fmax = problem._optimal_value
        list_fidelities = [cost,1.0]
        ninits = [10, 10]

    return ndim, problem, bounds, fmax, list_fidelities, ninits


def inference_regret(MFBO, model, fmax, cost_model, problem, ndim):
    xstar = MFBO.get_recommendation(model)
    xstar, trueval, _ = x_obj_cost(xstar, cost_model, problem, ndim)
    IR = fmax - trueval
    return float(IR)

def simple_regret(train_x, train_obj, fmax):
    samples = train_obj[is_primary_source(train_x)]
    SR = fmax - samples.max()
    return float(SR)

def augment_data(new_x, new_obj, train_x, train_obj, train_x_2=None, train_obj_2=None):
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    if train_x_2 is None:
        return train_x,train_obj
    else:
        train_x_2 = torch.cat([train_x_2, new_x])
        train_obj_2 = torch.cat([train_obj_2, new_obj])
        return train_x, train_obj, train_x_2, train_obj_2

def generate_initial_data(problem, ninits, bounds, ndim, list_fidelities, seed=None, method="random"):
    #ninits = list of ninit per fidelity
    ninit = sum(ninits)
    bounds_ = bounds[:,:-1]
    if seed is not None:
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
    if method=="random":
        train_x = torch.rand(ninit, bounds_.size(1))
    elif method=="latinhypercube":
        engine = LatinHypercube(d=ndim,seed=seed)
        train_x = torch.tensor(engine.random(n=ninits[0]))
        for i in range(len(list_fidelities)-1):
            j = i + 1
            train_x = torch.cat((train_x, torch.tensor(engine.random(n=ninits[j]))), dim=0)
    else:
        raise ValueError("Invalid initial training data sampling method")
    train_x = bounds_[0] + (bounds_[1] - bounds_[0]) * train_x
    train_fid = torch.t(torch.tensor([sum([[list_fidelities[i] for n in range(ninits[i])] for i in range(len(list_fidelities))],[])]))
    train_x_full = torch.cat((train_x, train_fid), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)
    return train_x_full, train_obj

def optimize_entropy(model, minORmax, problem, bounds, ndim, list_fidelities, niter=2):
    bounds__ = tuple([(bounds.tolist()[0][x], bounds.tolist()[1][x]) for x in range(ndim)] + [(1.0, 1.0)])
    if minORmax == "max": a = -1
    if minORmax == "min": a = 1
    fun = lambda x: a * float(model.posterior(torch.tensor(x).view(1, -1),observation_noise=True).mvn.entropy())
    fvals = []
    for i in range(niter):
        x0 = np.array(
            generate_initial_data(problem,[0 for x in range(len(list_fidelities) - 1)] + [1], bounds,ndim,list_fidelities,seed=0, method="random")[
                0]).reshape(-1, )
        res = minimize(fun, x0, bounds=bounds__, method='Nelder-Mead', options={'disp': False, 'maxiter': 10000})
        fvals.append(res.fun)
    return a * min(fvals)


def x_obj_cost(candidate, cost_model, problem, ndim):
    """Return """
    cost = cost_model(candidate).sum()
    new_x = candidate.detach().view(1,ndim+1)
    new_obj = problem(new_x).unsqueeze(-1)
    return new_x, new_obj, cost


def bin_mean_cost(li, budget, idx_comp=0, nbins=10):

    """Construct mean stds and mean cost from a list of tensors of different sizes."""

    bins = np.linspace(0, budget, nbins)
    out_bins = bins[1:] - (bins[1:] - bins[:-1]) / 2
    bins = list(bins)
    binned_vals = [[] for _ in range(nbins - 1)]
    for i in range(len(li)):
        cum_costs = np.cumsum(li[i][:, 2])
        idx = np.searchsorted(cum_costs, bins)
        for j in range(len(bins) - 1):
            app = list(li[i][idx[j]:idx[j+1], idx_comp])
            if len(app):
                binned_vals[j].append(app)
    binned_vals = [sum(b, []) for b in binned_vals]
    keep = [i for i in range(len(binned_vals)) if len(binned_vals[i])]
    out_bins = [out_bins[i] for i in range(len(out_bins)) if i in keep]
    binned_vals = [binned_vals[i] for i in keep]

    mean = np.array([np.mean(b) for b in binned_vals])
    std = np.array([1.96 * np.std(b) / np.sqrt(len(b)) for b in binned_vals])

    for i in range(len(mean)-1):
       if mean[i] < mean[i + 1] and not idx_comp:
           mean[i + 1]  = mean[i]
    return torch.tensor([mean]).t(), torch.tensor([std]).t(), out_bins


def custom_relevance_measure(model_mf, model_sf, train_x, x_m, x):
    mean_sf_m = float(torch.mean(model_sf.posterior(train_x[is_primary_source(train_x)][:, :-1]).mean))
    mean_mf_m = float(torch.mean(model_mf.posterior(train_x[is_primary_source(train_x)]).mean))
    cross_covar = model_mf.posterior(torch.vstack((x, x_m))).mvn.covariance_matrix
    mf_x_m = model_mf.posterior(x_m).mean
    mf_x_l = model_mf.posterior(x).mean
    sf_x_m = model_sf.posterior(x_m[:, :-1]).mean
    var_m_sf = model_sf.posterior(x_m[:, :-1]).variance
    #If var(x,l) is too high, we cannot trust mu(x,l) (with GP kernels, often then mu(x,l)-mu(x,m)=0)
    #For this reason, there is first if statemen. This modified version of signal
    #does not depend on mu(x,l)-mu(x,m), when var(x,l) is too high
    if cross_covar[0, 0] > 0.015: #roughly 25% of max standard deviation (0.5) in case of output normalization into [0.1]
        X = abs(sf_x_m - mf_x_m)
        sf_mf_covar = min((sf_x_m-mean_sf_m) * (mf_x_m-mean_mf_m), var_m_sf, cross_covar[0,0])
        threshold = (var_m_sf - 2 * sf_mf_covar + cross_covar[0, 0]) ** 0.5
        return X < threshold/4
    else:
        X = abs(sf_x_m - mf_x_m) * abs(mf_x_m - mf_x_l)
        X = X ** 0.5
        Xvar = cross_covar[0,0] - 2 * cross_covar[0, 1] + cross_covar[1, 1]
        sf_mf_covar = min((sf_x_m-mean_sf_m) * (mf_x_m-mean_mf_m), var_m_sf, cross_covar[0,0])
        Yvar = (var_m_sf - 2 * sf_mf_covar + cross_covar[0, 0])
        threshold = ((Xvar * Yvar / (Xvar + Yvar)) ** 0.5)
    return X < threshold


def dict_fidelities(dict, exp, method, cost, condvar, condig, mogp):

    """Return the right entry given the method."""
    # quick way to deal with dict generated before adding mogp param
    if "F" not in "".join(dict[exp]):
        if "SF" in method:
            out = dict[exp][mogp][method]
        elif ("rMF" not in method and "MF" in method):
            out = dict[exp][mogp][method][cost]
        else:
            out = dict[exp][mogp][method][cost][condvar][condig]
    else:
        if "SF" in method:
            out = dict[exp][method]
        elif ("rMF" not in method and "MF" in method):
            out = dict[exp][method][cost]
        else:
            out = dict[exp][method][cost][condvar][condig]
    return out


def plot_dict():

    """Matplotlib options."""

    return {'SF-MES': ['C2', 'o', 5],
            'MF-MES': ['violet', 's', 1],
            'rMF-MES': ['darkblue', 'x', 10],
            'SF-KG': ['greenyellow', 'o', 5],
            'MF-KG': ['purple', 's', 1],
            'rMF-KG': ['cyan', 'x', 10],
            'SF-GIBBON': ['lightgreen', 'o', 5],
            'MF-GIBBON': ['red', 's', 1],
            'rMF-GIBBON': ['blue', 'x', 10]
                }


class CostAdhoc(DeterministicModel):
    r"""
    For each (q-batch) element of a candidate set `X`, this module computes a
    cost of the form

        cost = adhoc

    Example:
        >>> from botorch.models import AffineFidelityCostModel
        >>> from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        >>> cost_model = CostOne(
        >>>    fidelity_weights={6: 1.0}
        >>> )
        >>> cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    """

    def __init__(
        self,
        fidelity_weights= None, # same
        fixed_cost: float = 0.0, # will not be used anyway
    ) -> None:
        r"""
        Args:
            fidelity_weights: A dictionary mapping a subset of columns of `X`
                (the fidelity parameters) to its associated weight in the
                affine cost expression. If omitted, assumes that the last
                column of `X` is the fidelity parameter with a weight of 1.0.
            fixed_cost: The fixed cost of running a single candidate point (i.e.
                an element of a q-batch).
        """
        if fidelity_weights is None:
            fidelity_weights = {-1: 1.0}
        super().__init__()
        self.fidelity_dims = sorted(fidelity_weights)
        self.fixed_cost = fixed_cost
        weights = torch.tensor([fidelity_weights[i] for i in self.fidelity_dims])
        self.register_buffer("weights", weights)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost on a candidate set X.

        Computes a cost of the form

            cost = fixed_cost + sum_j weights[j] * X[fidelity_dims[j]]

        for each element of the q-batch

        Args:
            X: A `batch_shape x q x d'`-dim tensor of candidate points.

        Returns:
            A `batch_shape x q x 1`-dim tensor of costs.
        """
        if X.ndim == 3:
            lin_cost = torch.zeros(len(X))
            for i in range(len(X)):
                lin_cost[i] = torch.tensor([X[i][0][-1]]) if X[i][0][-1] == 1 else torch.tensor([.2])
        if X.ndim == 2:
            lin_cost = torch.zeros(len(X))
            for i in range(len(X)):
                lin_cost[i] = torch.tensor([X[i][-1]]) if X[i][-1] == 1 else torch.tensor([.2])
        elif X.ndim == 1:
            lin_cost = torch.tensor(X[-1]) if X[-1] == 1 else torch.tensor(.2)
        return self.fixed_cost + lin_cost.unsqueeze(-1)


def build_results(nfiles, name):
    li = []
    for i in range(nfiles):
        li.append(torch.load(f'results/exp_{name}_{i}.pt'))
    torch.save(li, f'results/{name}_results.pt')

def getIS(x):
    if len(x.shape) == 2:
        return round(float(x[0, -1]), 5) #round up to 5 decimals
    else:
        return round(float(x[-1]), 5)  # round up to 5 decimals
