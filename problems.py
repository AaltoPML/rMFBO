from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.test_functions.multi_fidelity import AugmentedHartmann, AugmentedBranin

from misc import is_primary_source
from xgb_utils import DiabetesFunctional


class Hartmann(AugmentedHartmann):
    def __init__(self, noise_std: Optional[float] = None):
        super().__init__(noise_std=noise_std)
        self._max_ = 3.322367993160791
        self._min_ = 0.0
        self._optimal_value = 1.0
        self._pis = "hartmann"
        self._ais = "hartmann_low"
    def __call__(self, X: Tensor) -> Tensor:
        H = -super().evaluate_true(X)  #non-negated is not intresting objective
        H = (H - self._min_)/(self._max_-self._min_)  # normalize
        return torch.where(torch.tensor(is_primary_source(X)), H + self.noise_std * torch.randn_like(H) , H)


class HartmannCost():
    def __init__(self, noise_std: Optional[float] = None):
        self.noise_std = noise_std
        self._optimal_value = 1.0
        self._pis = "hartmann"
        self._ais = "hartmann_low"
        self.hartmann = Hartmann(self.noise_std)
        self.hartmannlow = FixedHartmann()
    def __call__(self, X: Tensor) -> Tensor:
        return torch.where(torch.tensor(is_primary_source(X)), self.hartmann(X), self.hartmannlow(X))


class Rosenbrock():
    def __init__(self, dim=2, noise_std: Optional[float] = None, negate: bool =False):
        self.negate = negate
        self._dim = dim
        self.noise_std = noise_std
        if not self.negate:
            self._min_ = -90036.0 * (self._dim - 1)
            self._max_ = 0.0
        else:
            self._min_ = 0.0
            self._max_ = 90036.0 * (self._dim - 1)
        self._optimal_value = 1.0
    def __call__(self, X: Tensor) -> Tensor:
        X_curr = X[..., :self._dim - 1]
        X_next = X[..., 1:self._dim]
        H = (
            -(
                100 * (X_next - X_curr ** 2) ** 2
                + (X_curr - 1 ** 2)**2
            )
        ).sum(dim=-1)
        if self.negate:
            H= - H
        H = (H - self._min_)/(self._max_-self._min_)
        return torch.where(torch.tensor(is_primary_source(X)), H + self.noise_std * torch.randn_like(H) , H)


class RosenbrockSinus():
    def __init__(self, noise_std: Optional[float] = None, negate: bool = False, dim=2):
        self.rosenbrock = Rosenbrock(dim=dim, noise_std=noise_std, negate=negate)
        self.sinus = Rosenbrock(dim=dim, noise_std=0, negate=negate)
        self._optimal_value = 1.0
        self._pis = "rosenbrock"
        self._ais = "rosenbrocksinus"
    def __call__(self, X: Tensor) -> Tensor:
        out = self.rosenbrock(X)
        H = torch.where(torch.tensor(is_primary_source(X)), 1- out, 1-(out + out.mean(dim=-1) * .8 * torch.sin(X[..., :-1].sum(dim=-1))))
        return H


class Ackley():
    def __init__(self, dim=2, noise_std: Optional[float] = None, negate: bool = False):
        self.negate = negate
        #### bounds in the branin case
        if not self.negate:
            # self._min_ = -10.5164
            self._min_ = -17.50447 # when used with branin
            self._max_ = 0.0
        if self.negate:
            self._min_ = 0.0
            # self._max_ = 10.5164
            self._max_ = 17.50447 #  when used with branin
        # self._min_ = - 15.7518
        # self._max_ = 0.0
        self._optimal_value = 1.0
        self._dim = dim
    def __call__(self, X: Tensor) -> Tensor:
        H = - (- 20 * torch.exp(- .2 * torch.sqrt(1. / 6 * (torch.sum(X[:, :-1]**2, axis = 1)))) - torch.exp((1./ self._dim * (torch.sum(torch.cos(2 * torch.pi * X[:, :-1]), axis = 1)))) + torch.exp(torch.tensor([1]))+20)
        if self.negate:
            H = - H
        return (H - self._min_)/(self._max_ - self._min_) # normalize


class CurrinExponential():
    def __init__(self, noise_std: Optional[float] = None, negate: bool = False):
        self.negate = negate
        self.noise_std = noise_std
        if not self.negate:
            self._min_ = 1.1804
            self._max_ = 13.7987
        else:
            self._min_ = - 13.7987
            self._max_ = - 1.1804
        self._optimal_value = 1.0
    def __call__(self, X: Tensor) -> Tensor:
        H = (1 - torch.exp(- 1 / (2 * X[:, 1]))) * (2300 * X[:, 0]**3 + 1900 * X[:, 0]**2 + 2092 * X[:, 0] + 60) / (100 * X[:, 0]**3 + 500 * X[:, 0]**2 + 4 * X[:, 0] + 20) 
        if self.negate:
            H = -H
        H = (H - self._min_)/(self._max_ - self._min_) # normalize
        return torch.where(torch.tensor(is_primary_source(X)), H + self.noise_std * torch.randn_like(H) , H)


class CurrinNegated():
    def __init__(self, noise_std: Optional[float] = None):
        self.currin = CurrinExponential(noise_std=noise_std, negate=False)
        self.negatedcurrin = CurrinExponential(noise_std=0, negate=True)
        self._optimal_value = 1.0
        self._pis = "currin"
        self._ais = "currinnegated"
    def __call__(self, X: Tensor) -> Tensor:
        H = torch.where(torch.tensor(is_primary_source(X)), self.currin(X), self.negatedcurrin(X))
        return H


class Branin(AugmentedBranin):
    def __init__(self,noise_std: Optional[float] = None, negate: bool = False):
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self._pis = "branin"
        self._ais = "branin_lowfid"
        if not self.negate:
            self._max_ = 308.12909601160663
            self._min_ = 0.3978873577401032
        else:
            self._min_ = -308.12909601160663
            self._max_ = -0.3978873577401032
        self._optimal_value = 1.0
    def __call__(self, X: Tensor) -> Tensor:
        H = super().evaluate_true(X)
        if self.negate:
            H = - H
        H = (H - self._min_)/(self._max_-self._min_)  # normalize
        return torch.where(torch.tensor(is_primary_source(X)), H + self.noise_std * torch.randn_like(H) , H)


class HartmannRosenbrock():
    def __init__(self, noise_std: Optional[float] = None):
        self.hartmann = Hartmann(noise_std=noise_std)
        self.rosenbrock = Rosenbrock(dim=6,noise_std=0, negate=True)
        self._optimal_value = 1.0
        self._pis = "hartmann"
        self._ais = "rosenbrock"
    def __call__(self, X: Tensor) -> Tensor:
        return torch.where(torch.tensor(is_primary_source(X)), self.hartmann(X), self.rosenbrock(X))


class HartmannMultiple():
    def __init__(self, noise_std: Optional[float] = None):
        self.hartmann = Hartmann(noise_std=noise_std)
        self.rosenbrock = Rosenbrock(dim=6,noise_std=0, negate=True)
        self._optimal_value = 1.0
        self._pis = "hartmann"
        self._ais = "hartmannlow0.8/hartmannlow0.1/rosenbrock"
    def __call__(self, X: Tensor) -> Tensor: 
        return torch.where(torch.tensor(is_primary_source(X, opposite=True)), self.rosenbrock(X), self.hartmann(X))


class BraninMultiple():
    def __init__(self, noise_std: Optional[float] = None):
        self.branin = Branin(noise_std=noise_std)
        self.ackley = Ackley(dim=2,noise_std=0, negate=True)
        self._optimal_value = 1.0
        self._pis = "branin"
        self._ais = "braninlow0.8/braninlow0.1/ackley"
    def __call__(self, X: Tensor) -> Tensor: 
        return torch.where(torch.tensor(is_primary_source(X, opposite=True)), self.ackley(X), self.branin(X))


class XGB:
    def __init__(self, noise_std: Optional[float] = None):
        self._pis = "XGB_100boosters"
        self._ais = "XGB_10boosters"
        self._optimal_value = 1.0
        self.target = DiabetesFunctional(100)
        self.low = DiabetesFunctional(10)
        self._max_ = - 0.6591 # computed over 30000 uniform samples...
        self._min_ = - 2.2039 # same
    def __call__(self, X: Tensor) -> Tensor:
        H = torch.where(torch.tensor(is_primary_source(X)), self.target(X[:, :-1]), self.low(X[:, :-1]))
        return (H - self._min_) / (self._max_ - self._min_)  # normalize


class FixedHartmann(SyntheticTestFunction):
    r"""Augmented Hartmann synthetic test function.

    7-dimensional function (typically evaluated on `[0, 1]^7`), where the last
    dimension is the fidelity parameter.

        H(x) = -(ALPHA_1 - 0.1 * (1-x_7)) * exp(- sum_{j=1}^6 A_1j (x_j - P_1j) ** 2) -
            sum_{i=2}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij) ** 2)

    H has a unique global minimizer
    `x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0]`

    with `H_min = -3.32237`
    """

    dim = 7
    _bounds = [(0.0, 1.0) for _ in range(7)]
    _optimal_value = -3.32237
    _optimizers = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0)]
    _check_grad_at_opt = False

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        A = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        P = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))

        self._max_ = 3.322367993160791
        self._min_ = 0.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(
            self.A * (X[..., :6].unsqueeze(1) - 0.0001 * self.P) ** 2, dim=2
        )
        alpha1 = self.ALPHA[0] - 0.1 * (0.8)
        H = -(
            -torch.sum(self.ALPHA[1:] * torch.exp(-inner_sum)[..., 1:], dim=1)
            - alpha1 * torch.exp(-inner_sum)[..., 0]
        )
        return (H - self._min_)/(self._max_-self._min_)  # normalize