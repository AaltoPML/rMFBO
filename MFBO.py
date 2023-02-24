import torch
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition import PosteriorMean
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition import qLowerBoundMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy, qMaxValueEntropy, qMultiFidelityMaxValueEntropy, qMultiFidelityKnowledgeGradient, qKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch import fit_gpytorch_model
from torchquad import MonteCarlo

from misc import is_primary_source


class MFBO:

    def __init__(self, ndim, list_fidelities, bounds,
                 cost_model=None, cost_aware_utility=None,
                 cost_ig=None, cost_aware_ig=None, jointmogp="downsampling"):

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        self.ndim = ndim
        self.list_fidelities = list_fidelities
        self.target_fidelities = {ndim: list_fidelities[-1]}
        self.fidelities = torch.tensor(list_fidelities, **self.tkwargs)
        self.bounds = bounds
        self.jointmogp = jointmogp

        if len(self.list_fidelities) == 2:
            global SingleTaskMultiFidelityGPMiso
            from misokernel.gp_regression_fidelity_miso_numAIS1 import SingleTaskMultiFidelityGPMiso
        elif len(self.list_fidelities) == 4:
            global SingleTaskMultiFidelityGPMiso
            from misokernel.gp_regression_fidelity_miso_numAIS3 import SingleTaskMultiFidelityGPMiso

        #grid
        self.ngrid = 1000
        bounds_ = self.bounds[:, :-1]
        self.candidate_set = torch.rand(self.ngrid, bounds_.size(1))
        self.candidate_set = bounds_[0] + (bounds_[1] - bounds_[0]) * self.candidate_set
        fid = torch.t(torch.tensor([[list_fidelities[-1] for _ in range(self.ngrid)]]))
        self.candidate_set = torch.cat((self.candidate_set, fid), dim=1)

        #optim specs
        self.NUM_RESTARTS = 5
        self.RAW_SAMPLES = 128
        self.NUM_FANTASIES = 128

        self.cost_model = cost_model
        self.cost_aware_utility = cost_aware_utility
        self.cost_ig = cost_ig
        self.cost_aware_ig = cost_aware_ig

        self.mc = MonteCarlo()


    def initialize_model(self, train_x, train_obj):

        ''' What kernel to use is MOGP model? '''

        if torch.equal(train_x[:, -1], torch.ones(len(train_x), **self.tkwargs)):
            if self.jointmogp == "downsampling":
                model = SingleTaskGP(train_x[:, :-1], train_obj,
                                     covar_module=ScaleKernel(RBFKernel(ard_num_dims=self.ndim, lengthscale_prior=GammaPrior(3.0, 6.0)),
                                                              outputscale_prior=GammaPrior(2.0, 0.15)), # same priors as STMFGP
                                     outcome_transform=Standardize(m=1))
            elif self.jointmogp == "lineartruncated" or "miso":
                model = SingleTaskGP(train_x[:, :-1], train_obj,
                     covar_module=ScaleKernel(MaternKernel(ard_num_dims=self.ndim, lengthscale_prior=GammaPrior(3.0, 6.0)),
                                              outputscale_prior=GammaPrior(2.0, 0.15)),outcome_transform=Standardize(m=1))
        else:
            if self.jointmogp == "downsampling":
                model = SingleTaskMultiFidelityGP(train_x, train_obj, linear_truncated=False,
                                                  outcome_transform=Standardize(m=1), data_fidelity=self.ndim)
            elif self.jointmogp == "lineartruncated":
                model = SingleTaskMultiFidelityGP(train_x, train_obj, linear_truncated=True,
                                  outcome_transform=Standardize(m=1), data_fidelity=self.ndim)
            elif self.jointmogp == "miso":
                model = SingleTaskMultiFidelityGPMiso(train_x, train_obj, linear_truncated=False, miso=True,
                                      outcome_transform=Standardize(m=1), data_fidelity=self.ndim)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model


    def optimize_alpha(self,model,acquisitionfunc,iss=None):
        # iss = (list) information sources that are take into account in the acquisition function optimization
        if acquisitionfunc == "MF-MES":
            alpha = self.get_mfmes(model)
            new_x, _ = self.optimize_mfmes(alpha,iss)
        elif acquisitionfunc == "SF-MES":
            if model.__class__.__name__ == "SingleTaskGP":
                alpha = self.get_mes(model)
            else:
                alpha = self.get_mes_last_iterate(model)
            new_x, _ = self.optimize_mes(alpha)
        elif acquisitionfunc == "MF-KG":
            alpha = self.get_mfkg(model)
            new_x, _ = self.optimize_mfkg(alpha,iss)
        elif acquisitionfunc == "SF-KG":
            # if model.__class__.__name__ == "SingleTaskGP":
            # #   alpha = self.get_kg(model)
            # else:
            #     alpha = self.get_kg_last_iterate(model)
            alpha = self.get_kg(model)
            new_x, _ = self.optimize_kg(alpha,mf_at_pis=False)
        elif acquisitionfunc == "MF-GIBBON":
            alpha = self.get_mfgibbon(model)
            new_x, _ = self.optimize_mfgibbon(alpha,iss)
        elif acquisitionfunc == "SF-GIBBON":
            if model.__class__.__name__ == "SingleTaskGP":
                alpha = self.get_gibbon(model)
            else:
                alpha = self.get_gibbon_last_iterate(model)
            new_x, _ = self.optimize_gibbon(alpha)
        else:
            raise ValueError("Acquisition function not recognized")
        return new_x

    def project(self,X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    ''' Functions for MF-MES and SF-MES '''

    def get_mfmes(self,model):
        return qMultiFidelityMaxValueEntropy(
            model=model,
            num_fantasies=self.NUM_FANTASIES,
            cost_aware_utility=self.cost_aware_utility,
            project=self.project,
            candidate_set=self.candidate_set,
        )

    def get_mes(self,model):
        return qMaxValueEntropy(model=model, candidate_set=self.candidate_set[:, :-1])


    def get_mes_last_iterate(self,model):
        return FixedFeatureAcquisitionFunction(
            acq_function=qMaxValueEntropy(model=model, candidate_set=self.candidate_set),
            d=self.ndim+1,
            columns=[self.ndim],
            values=[self.list_fidelities[-1]],
        )

    def optimize_mfmes(self,mes_acqf,iss):
        # generate new candidates
        if iss is None: iss = self.list_fidelities
        candidates, MES = optimize_acqf_mixed(
            acq_function=mes_acqf,
            bounds=self.bounds,
            fixed_features_list=[{self.ndim: f} for f in iss],
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates, MES


    def optimize_mes(self,mes_acqf):
        candidates, MES = optimize_acqf(
            acq_function=mes_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        # add the fidelity parameter
        candidates = torch.cat((candidates, torch.ones(1).unsqueeze(-2)), dim=1)
        return candidates, MES

    ''' Functions for MF-KG and SF-KG '''

    def get_mfkg(self, model):
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=self.ndim+1,
            columns=[self.ndim],
            values=[self.list_fidelities[-1]],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=10,
            raw_samples=1024,
            options={"batch_limit": 10, "maxiter": 200},
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.NUM_FANTASIES,
            current_value=current_value,
            cost_aware_utility=self.cost_aware_utility,
            project=self.project,
        )

    def optimize_mfkg(self,kg_acqf,iss):
        if iss is None: iss = self.list_fidelities
        candidates, kg = optimize_acqf_mixed(
            acq_function=kg_acqf,
            bounds=self.bounds,
            fixed_features_list=[{self.ndim: f} for f in iss],
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates, kg

    def get_kg(self, model):
        return qKnowledgeGradient(model=model, num_fantasies=self.NUM_FANTASIES)

    def get_kg_last_iterate(self, model):
        return FixedFeatureAcquisitionFunction(
            acq_function=qKnowledgeGradient(model=model, num_fantasies=self.NUM_FANTASIES),
            d=self.ndim+1,
            columns=[self.ndim],
            values=[self.list_fidelities[-1]],
        )

    def optimize_kg(self, kg_acqf, mf_at_pis=False):
        candidates, kg = optimize_acqf(
            acq_function=kg_acqf,
            bounds=self.bounds[:, :-1],
            q=(1 if not mf_at_pis else self.NUM_FANTASIES+1), #Ad-hoc fix for KG-error when get_kg_last_iterate
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        if mf_at_pis:
            candidates = candidates[0,:].unsqueeze(0)
        # add the fidelity parameter
        candidates = torch.cat((candidates, torch.ones(1).unsqueeze(-2)), dim=1)
        return candidates, kg

    """Gibbon functions"""

    def get_mfgibbon(self,model):
        return qMultiFidelityLowerBoundMaxValueEntropy(
            model=model,
            num_fantasies=self.NUM_FANTASIES,
            cost_aware_utility=self.cost_aware_utility,
            project=self.project,
            candidate_set=self.candidate_set,
        )

    def get_gibbon(self,model):
        return qLowerBoundMaxValueEntropy(model=model, candidate_set=self.candidate_set[:, :-1])

    def get_gibbon_last_iterate(self,model):
        return FixedFeatureAcquisitionFunction(
            acq_function=qLowerBoundMaxValueEntropy(model=model, candidate_set=self.candidate_set),
            d=self.ndim+1,
            columns=[self.ndim],
            values=[self.list_fidelities[-1]],
        )

    def optimize_mfgibbon(self,gibbon_acqf,iss):
        if iss is None: iss = self.list_fidelities
        # generate new candidates
        candidates, gibbon = optimize_acqf_mixed(
            acq_function=gibbon_acqf,
            bounds=self.bounds,
            fixed_features_list=[{self.ndim: f} for f in iss],
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates, gibbon

    def optimize_gibbon(self,gibbon_acqf):
        candidates, gibbon = optimize_acqf(
            acq_function=gibbon_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        # add the fidelity parameter
        # candidates = gibbon_acqf._construct_X_full(candidates)
        candidates = torch.cat((candidates, torch.ones(1).unsqueeze(-2)), dim=1)
        return candidates, gibbon

    """ Auxiliary function to get maximizer of predictive mean """

    def get_recommendation(self,model):
        if model.__class__.__name__ == "SingleTaskGP": # very ugly... :'(
            rec_acqf = PosteriorMean(model)
        else:
            rec_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(model),
                d=self.ndim+1,
                columns=[self.ndim],
                values=[1],
            )
        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        final_rec = rec_acqf._construct_X_full(final_rec) if model.__class__.__name__ != "SingleTaskGP" else torch.cat((final_rec, torch.ones(1).unsqueeze(-2)), dim=1)
        return final_rec

    """ Functions for robust MFBO (MES)"""

    def cost(self,l):
        x = torch.tensor([0,]*self.ndim + [l])
        cost = self.cost_model(x).sum()
        return float(cost)


    def botorch_IG(self, x, model,MF=True):
        """ Returns information gain of input-IS pair (x,l) given a model where l is given by last entry of x."""
        if MF:
            use_mfmes = qMultiFidelityMaxValueEntropy(
                model=model,
                num_fantasies=1024,  #earlier 128, TODO: experiment with this
                cost_aware_utility=self.cost_aware_ig,
                project=self.project,
                candidate_set=self.candidate_set,
            )
            return use_mfmes(x.unsqueeze(-2))
        else:
            use_mfmes = qMaxValueEntropy(model=model, candidate_set=self.candidate_set[:, :-1])
            return use_mfmes(x.unsqueeze(-2))


    def optimal_irmodel(self,model_sf,model_mf,train_x,train_obj):
        xstar = train_x[torch.argmax(train_obj)].view(1, -1)
        irmodel = model_sf if abs(model_sf.posterior(xstar[:, :-1]).mean - torch.max(train_obj)) < abs(
        model_mf.posterior(xstar).mean - torch.max(train_obj)) else model_mf
        return irmodel

    def nearest_neighbor(self,x, train_x, train_obj):
        samples_y = train_obj[is_primary_source(train_x)]
        samples_x = train_x[is_primary_source(train_x)]
        nn_x = samples_x[torch.argmin(torch.cdist(samples_x, x, p=2.0))]
        nn_y = samples_y[torch.argmin(torch.cdist(samples_x, x, p=2.0))]
        return nn_x.view(1, -1), nn_y

    def best_pseudo_observation(self,x, model_mf,model_sf, train_x, train_obj):
        x = x.view(1,-1)
        nn_x, nn_y = self.nearest_neighbor(x, train_x, train_obj)
        mu_sf = model_sf.posterior(torch.vstack((nn_x[:, :-1],x[:, :-1])).unsqueeze(-2)).mean
        mu_mf = model_mf.posterior(torch.vstack((nn_x,x)).unsqueeze(-2)).mean
        if torch.abs(mu_sf[0] - nn_y) < torch.abs(mu_mf[0] - nn_y):
            return float(mu_sf[1])
        else:
            return float(mu_mf[1])

    def update_pseudo_samples(self,train_x_sf,train_obj_sf,train_x,train_obj,model,model_sf,psample_indices):
        for i,psample in enumerate(psample_indices):
            if psample==1:
                x=train_x_sf[i,:]
                train_obj_sf[i,0] = self.best_pseudo_observation(x, model,model_sf, train_x, train_obj)
        return train_obj_sf

    def update_model(self,train_x,train_obj):
        mll, model = self.initialize_model(train_x, train_obj)
        fit_gpytorch_model(mll)
        return model

    def mean_IG(self,train_x,train_obj):
        model_true_sf = self.update_model(train_x[is_primary_source(train_x)], train_obj[is_primary_source(train_x)])
        bounds = [[float(self.bounds[0,i]),float(self.bounds[1,i])] for i in range(self.ndim)]
        mean_ig = self.mc.integrate(
            lambda x : torch.clamp(self.botorch_IG(x, model_true_sf, MF=False),min=0),
            dim=self.ndim,
            N=3000,
            integration_domain=bounds,
            backend="torch",
        )
        volume = 1
        for i in range(len(bounds)):
            side_length = bounds[i][1] - bounds[i][0]
            volume = volume * side_length
        mean_ig = mean_ig.item() / volume
        return mean_ig


