import os
import numpy as np
import warnings
import torch
torch.multiprocessing.set_sharing_strategy('file_system') #for RuntimeError in Triton: "received 0 items of ancdata"
import functools
import multiprocessing as mp
from scipy.optimize import OptimizeWarning

from MFBO import MFBO as mfbo
from misc import augment_data, build_combinations, CostAdhoc, CostOne, generate_initial_data, get_problem_settings, inference_regret, is_primary_source, parser_bo, simple_regret, x_obj_cost, getIS


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
torch.set_default_tensor_type(torch.DoubleTensor) #otherwise mixed floats and doubles cause a lot of headache

warnings.filterwarnings("ignore", category=UserWarning) # damn torch triangular matrices warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)


# Detect the number of CPUs we have available.  If in slurm, use the SLURM_CPUS_PER_TASK environment variable which Slurm lets.
if 'SLURM_CPUS_PER_TASK' in os.environ:
    cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    print("Dectected %s CPUs through slurm"%cpus)
else:
    cpus = 6 #os.cpu_count() #or specify manually how many CPUs
    print("Running on default number of CPUs (default: all=%s)"%cpus)


#COSTS
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
FIXED_COST = 0 #fixed cost added to any fidelity
cost_model = AffineFidelityCostModel(fidelity_weights=None, fixed_cost=FIXED_COST)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
# IG Cost model loading as we will use the same through all runs.
IG_COST = CostOne(fixed_cost=0.0)
IG_INV_COST = InverseCostWeightedUtility(cost_model=IG_COST)

from multiprocessing import get_context
p = get_context("fork").Pool(cpus)

def eval_model(combi, BUDGET,
               save=None, verbose=False):

    experiment, method, cost, cond_var, cond_ig, mogp, repseed, ite = combi
    _, seed = repseed
    ndim, problem, bounds, fmax, list_fidelities, ninits = get_problem_settings(experiment, cost)
    m = list_fidelities[-1]
    lambda_m = x_obj_cost(torch.ones(ndim+1), cost_model, problem, ndim)[2].item()
    torch.manual_seed(seed)
    MFBO = mfbo(ndim, list_fidelities, bounds, cost_model, cost_aware_utility, IG_COST, IG_INV_COST, mogp)
    train_x_init, train_obj_init = generate_initial_data(problem, ninits, bounds, ndim, list_fidelities, seed=seed, method="latinhypercube")
    results = torch.zeros([0, 3])
    algo, acquisitionfunc = method.split("-", 1)

    if algo=='rMF':
        #ROBUST MFBO LOOP
        train_x, train_obj = torch.clone(train_x_init), torch.clone(train_obj_init)
        train_x_sf, train_obj_sf = torch.clone(train_x)[is_primary_source(train_x)], torch.clone(train_obj)[is_primary_source(train_x)]
        if cond_ig=="adaptive": #c2(t) = 100 * meanIG(t)
            lic = []
        else:
            c2 = float(cond_ig)
        #Start BO loop
        budgetleft = BUDGET
        t = 1
        model_mf = MFBO.update_model(train_x, train_obj)
        model_sf = MFBO.update_model(train_x_sf, train_obj_sf)
        ir_init = inference_regret(MFBO, model_mf, fmax, cost_model, problem, ndim)
        psample_indices = [0,]*ninits[-1]
        while np.floor(budgetleft/lambda_m) >= 2:
            print("round: " + str(t)) if (not t % 5 and verbose) else None
            x_sf = MFBO.optimize_alpha(model_sf, 'SF-'+acquisitionfunc)
            sigma = torch.sqrt(model_mf.posterior(x_sf).variance).item()
            condition1 = (sigma <= cond_var)
            condition2 = False
            if condition1:
                x_mf = MFBO.optimize_alpha(model_mf, 'MF-' + acquisitionfunc)
                if is_primary_source(x_mf):
                    condition2 = True
                else:
                    iss_left = set(list_fidelities)
                    iss_left.remove(m)
                    if cond_ig=="adaptive":
                        mean_ig = MFBO.mean_IG(train_x,train_obj)
                        lic.append([mean_ig, t])
                        c2 = 100 * mean_ig
                    while not condition2 and len(iss_left)>0:
                        l = getIS(x_mf)
                        cost_l = x_obj_cost(x_mf, cost_model, problem, ndim)[2].item()
                        IG_l = max([MFBO.botorch_IG(x_mf, model_mf).item(),0])
                        if IG_l / cost_l > c2:
                            condition2 = True
                        else:
                            iss_left.remove(l)
                            if len(iss_left)>0: x_mf = MFBO.optimize_alpha(model_mf, 'MF-' + acquisitionfunc, list(iss_left))

            if verbose: print("sigma_mf(x_sf,m) = " + str(sigma))
            if condition1 and condition2:
                _, new_obj, cost = x_obj_cost(x_mf, cost_model, problem, ndim)
                train_x, train_obj = augment_data(x_mf, new_obj, train_x, train_obj)
                if verbose:
                    model_mf = MFBO.update_model(train_x, train_obj)
                    print("next pseudo-query: " + str(x_sf))
                    print("next query: " + str(x_mf))
                    print("observation: " + str(new_obj.item()))
                    print("pseudo-query value " +str(model_mf.posterior(x_sf).mean.item()))
                    print("true pseudo-query value " + str(x_obj_cost(x_sf, cost_model, problem, ndim)[1].item()))
                #Initialize pseudo-data with value 0 for later to be updated in MFBO.update_pseudo_samples
                train_x_sf, train_obj_sf = augment_data(x_sf,torch.tensor([0]).unsqueeze(0),train_x_sf, train_obj_sf)
                psample_indices.append(1)
            else:
                _, new_obj, cost = x_obj_cost(x_sf, cost_model, problem, ndim)
                if verbose:
                    print("next query: " + str(x_sf))
                    print("observation: " + str(new_obj.item()))
                train_x, train_obj, train_x_sf, train_obj_sf = augment_data(x_sf, new_obj, train_x, train_obj, train_x_sf,train_obj_sf)
                psample_indices.append(0)
            budgetleft -= cost.item()
            print("cost: " + str(cost.item())) if verbose else None
            model_mf = MFBO.update_model(train_x, train_obj)
            train_obj_sf = MFBO.update_pseudo_samples(train_x_sf,train_obj_sf,train_x,train_obj,model_mf,model_sf,psample_indices)
            model_sf = MFBO.update_model(train_x_sf, train_obj_sf)
            """Regrets"""
            sr = simple_regret(train_x, train_obj, fmax)
            ir = inference_regret(MFBO, MFBO.optimal_irmodel(model_sf, model_mf, train_x, train_obj), fmax, cost_model,problem, ndim)
            if verbose:
                print("sr: " + str(sr))
                print("ir: " + str(ir))
            results = torch.cat([results, torch.tensor([[sr, ir, cost]])])
            t += 1
        ''' Last query: If Bayes-optimal x is not yet queried, then query it. Otherwise, next MFBO query at target fidelity.'''
        x_last = MFBO.get_recommendation(MFBO.optimal_irmodel(model_sf,model_mf,train_x,train_obj))
        if any([torch.allclose(x_last, train_x_sf[i,:]) for i in range(train_x_sf.shape[0])]):
            x_last = MFBO.optimize_alpha(model_mf, 'SF-'+acquisitionfunc)
        _, new_obj, cost = x_obj_cost(x_last, cost_model, problem, ndim)
        budgetleft -= cost
        train_x, train_obj, train_x_sf, train_obj_sf = augment_data(x_last, new_obj, train_x, train_obj, train_x_sf,train_obj_sf)
        model_mf = MFBO.update_model(train_x, train_obj)
        model_sf = MFBO.update_model(train_x_sf, train_obj_sf)
        """Regrets"""
        sr = simple_regret(train_x, train_obj, fmax)
        ir = inference_regret(MFBO, MFBO.optimal_irmodel(model_sf,model_mf,train_x,train_obj), fmax, cost_model, problem, ndim)
        if verbose:
            print("last query: " + str(x_last))
            print("last observation: " + str(new_obj.item()))
            print("sr: " + str(sr))
            print("ir: " + str(ir))
        results = torch.cat([results, torch.tensor([[sr, ir, cost]])])
        # if cond_ig=="adaptive": torch.save(lic, f'{save}_{ite}_cc.pt')

    if algo=='MF' or algo=='SF':
        if algo=='SF':
            train_x, train_obj = torch.clone(train_x_init)[is_primary_source(train_x_init)], torch.clone(train_obj_init)[is_primary_source(train_x_init)]
            train_x_init_save = torch.clone(train_x_init)[is_primary_source(train_x_init)],
            train_obj_init_save = torch.clone(train_obj_init)[is_primary_source(train_x_init)]
        if algo=='MF':
            train_x, train_obj = torch.clone(train_x_init), torch.clone(train_obj_init)
        budgetleft = BUDGET
        t = 1
        while np.floor(budgetleft/lambda_m) >= 1:
            print("round: " + str(t)) if (not t % 5 and verbose) else None
            model = MFBO.update_model(train_x, train_obj)
            if t == 1:
                ir_init=inference_regret(MFBO, model, fmax, cost_model, problem, ndim)
            new_x = MFBO.optimize_alpha(model, algo+'-'+acquisitionfunc)
            new_x, new_obj, cost = x_obj_cost(new_x, cost_model, problem, ndim)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            budgetleft -= cost.item()
            sr = simple_regret(train_x, train_obj, fmax)
            ir = inference_regret(MFBO, model, fmax, cost_model, problem, ndim)
            if verbose:
                print("next query: " + str(new_x))
                print("observation: " + str(new_obj.item()))
                print("cost: " + str(cost.item()))
                print("sr: " + str(sr))
                print("ir: " + str(ir))
            results = torch.cat([results, torch.tensor([[sr, ir, cost]])])
            t += 1

    if "MF" in method:
        train_x_init_save = train_x_init[:sum(ninits)]
        train_obj_init_save = train_obj_init[:sum(ninits)]
    sr_init = simple_regret(train_x_init[:sum(ninits)],train_obj_init[:sum(ninits)], fmax)
    results = torch.cat((torch.tensor([sr_init, ir_init, 0]).unsqueeze(-2), results))
    return results, combi, train_x_init_save, train_obj_init_save


def parallel_eval(combi, BUDGET, save, verbose, x):
    return eval_model(combi[x], BUDGET, save, verbose)


def main(save, N_REP, B, methods, seed, verbose, cond_var, cond_ig, experiments, costs, jointmogp, **args):

    torch.manual_seed(seed)
    combi = build_combinations(N_REP, experiments, costs, methods, cond_var, cond_ig, jointmogp, seed)
    selected_pool = mp.Pool(processes=cpus)
    with selected_pool as p:
        RES = p.map(functools.partial(parallel_eval, combi, B, save, verbose), range(len(combi)))
    p.close()
    torch.save(RES, f"{save}_results.pt")

if __name__ == "__main__":
    parser = parser_bo()
    main(**vars(parser.parse_args()))
