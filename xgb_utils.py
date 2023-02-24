import numpy as np
from sklearn import ensemble
import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch

def eval_rmse_score(pred, ground):
    rmse = np.sqrt(np.mean(np.square(pred-ground)))
    return rmse

def eval_accuracy_score(pred, ground):
    pred_labels = np.argmax(pred, axis=1)
    correct = np.sum(pred_labels==ground.squeeze())
    total = pred_labels.size
    return correct/total

def eval_auc_score(pred, ground):
    probs = softmax(pred, axis=1)
    label = ground.squeeze()
    auc = roc_auc_score(label, probs[:,1])
    return auc


def xgboost_regressor_binary_decoder_v2(X):
    
    X_np = np.array(X)

    if X_np.ndim == 2:
        X_np = np.squeeze(X_np)

    log_alpha = X_np[0]
    log_ccpa = X_np[1]
    log_subsample = X_np[2]
    log_max_features = X_np[3]
    log_learning_rate = X_np[4]
    
    alpha = np.power(10, log_alpha)
    if alpha >= 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(alpha))
        alpha=1.0-1e-3
    
    ccpa = np.power(10, log_ccpa)
    
    subsample = np.power(10, log_subsample)
    if subsample > 1.0:
        warnings.warn("subsample larger than 1.0, val = "+str(subsample))
        subsample=1.0
    
    max_features = np.power(10, log_max_features)
    if max_features > 1.0:
        warnings.warn("max_features larger than 1.0, val = "+str(max_features))
        max_features=1.0

    learning_rate = np.power(10, log_learning_rate)
    if learning_rate > 1.0:
        warnings.warn("max_features larger than 1.0, val = "+str(learning_rate))
        learning_rate=1.0
    
    xgb_config = {}
    xgb_config['alpha'] = alpha
    xgb_config['ccpa'] = ccpa
    xgb_config['subsample'] = subsample
    xgb_config['max_features'] = max_features
    xgb_config['learning_rate'] = learning_rate
    
    return xgb_config


def wrap_xgb_regressor_params(xgb_config, n_boosters):
    
    params = {'n_estimators': n_boosters,
              'loss': 'huber',
              'alpha': xgb_config['alpha'],
              'ccp_alpha': xgb_config['ccpa'],
              'subsample': xgb_config['subsample'],
              'max_features': xgb_config['max_features'],
              'criterion': 'friedman_mse',
              'learning_rate': xgb_config['learning_rate'],
              'n_iter_no_change':10}
    
    return params


def eval_xgb_regressor_performance(domain, designs, max_boosters):
    #if X.ndim == 2:
     #   X = X.squeeze()
        
    Xtr, ytr = domain.get_data(train=True, normalize=True)
    Xte, _ = domain.get_data(train=False, normalize=True)
    score = torch.zeros(len(designs))
    for i, x in enumerate(designs):
        xgb_config = xgboost_regressor_binary_decoder_v2(x)

        params = wrap_xgb_regressor_params(xgb_config, max_boosters)
        
        xgb_regressor = ensemble.GradientBoostingRegressor(**params)
        xgb_regressor.fit(Xtr, ytr.squeeze())
        
        pred = xgb_regressor.predict(Xte)
        score[i] = domain.metric(pred)
            
    return score

    
class DiabetesDomain:
    def __init__(self, partition_ratio, partition_seed):
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            X, y, test_size=partition_ratio, random_state=partition_seed)

        self.ytr = self.ytr.reshape([-1,1])
        self.yte = self.yte.reshape([-1,1])

        
    def get_data(self,train=True, normalize=False):
        if train:
            X = self.Xtr
            y = self.ytr
        else:
            X = self.Xte
            y = self.yte
        
        if normalize:
            scaler_X = preprocessing.StandardScaler().fit(self.Xtr)
            scaler_y = preprocessing.StandardScaler().fit(self.ytr)
            
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)

        return X, y
    
    def metric(self, pred, normalize=True, torch_tensor=False):
        if pred.ndim == 1:
            pred = pred.reshape([-1,1])
        
        if torch_tensor:
            pred = pred.data.cpu().numpy()
        
        scaler_y = preprocessing.StandardScaler().fit(self.ytr)
        if normalize:
            pred = scaler_y.inverse_transform(pred)
        
        score = eval_rmse_score(pred, self.yte)        
        score = score / scaler_y.scale_
        return - float(score)


def DiabetesFunctional(max_iters):
    domain = DiabetesDomain(partition_ratio=0.33, partition_seed=27)
    def func(binary_code):
        score = eval_xgb_regressor_performance(domain, binary_code, max_iters)
        return score
    return func