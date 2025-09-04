library(reticulate)

py_run_string("
import torch, gpytorch, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
torch.set_default_dtype(torch.float32)
np.random.seed(XXXX); torch.manual_seed(XXXX)

train_df = pd.read_csv('data.csv')
test_df  = pd.read_csv('vad.csv')
y = (train_df['Result'].values).astype(np.float32)
X = train_df.iloc[:, 1:].values
X_test_all = test_df.iloc[:, 1:].values

class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, hidden=64, out_dim=32, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class DeepKernelGP(gpytorch.models.ApproximateGP):
    def __init__(self, feat_dim, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim))
    def forward(self, x):
        mean = self.mean_module(x)
        cov  = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

def train_one_fold(X_tr, y_tr, X_va, y_va, cfg, max_epochs=50, batch_size=64, patience=7):
    scaler = StandardScaler().fit(X_tr)
    XT = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    XV = torch.tensor(scaler.transform(X_va), dtype=torch.float32)
    yT = torch.tensor(y_tr, dtype=torch.float32)
    yV = torch.tensor(y_va, dtype=torch.float32)

    feat = FeatureExtractor(XT.shape[1], hidden=cfg['hidden'], out_dim=cfg['feat_dim'], dropout=cfg['dropout'])
    idx = torch.randperm(XT.shape[0])[:cfg['m']]
    with torch.no_grad(): Z = feat(XT[idx])
    model = DeepKernelGP(cfg['feat_dim'], inducing_points=Z.clone())
    lik = gpytorch.likelihoods.BernoulliLikelihood()

    model.train(); lik.train()
    opt = torch.optim.Adam([{'params': feat.parameters()},{'params': model.parameters()}], lr=cfg['lr'], weight_decay=cfg['wd'])
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=yT.numel())
    loader = DataLoader(TensorDataset(XT, yT), batch_size=batch_size, shuffle=True)

    best_auc, wait = -1.0, 0
    for epoch in range(max_epochs):
        for xb, yb in loader:
            opt.zero_grad(); fb = model(feat(xb)); loss = -mll(fb, yb); loss.backward(); opt.step()
        model.eval(); lik.eval()
        with torch.no_grad(): pv = lik(model(feat(XV))).probs.cpu().numpy()
        auc = roc_auc_score(y_va, pv)
        model.train(); lik.train()
        if auc > best_auc + 1e-4: best_auc, wait = auc, 0
        else:
            wait += 1
            if wait >= patience: break
    return best_auc

def cv_mean_auc(X, y, cfg, n_splits=5, seed=2025):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in kf.split(X, y):
        aucs.append(train_one_fold(X[tr], y[tr], X[va], y[va], cfg))
    return float(np.mean(aucs))

def fit_full_and_predict(X_full, y_full, X_test, cfg, max_epochs=60, batch_size=64, patience=10, val_size=0.15, seed=2025):
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=val_size, random_state=seed, stratify=y_full)

    scaler = StandardScaler().fit(X_tr)
    XT  = torch.tensor(scaler.transform(X_tr),  dtype=torch.float32)
    XV  = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    XTe = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    yT  = torch.tensor(y_tr, dtype=torch.float32)
    yV  = torch.tensor(y_val, dtype=torch.float32)

    feat = FeatureExtractor(XT.shape[1], hidden=cfg['hidden'], out_dim=cfg['feat_dim'], dropout=cfg['dropout'])
    idx = torch.randperm(XT.shape[0])[:cfg['m']]
    with torch.no_grad(): Z = feat(XT[idx])
    model = DeepKernelGP(cfg['feat_dim'], inducing_points=Z.clone())
    lik = gpytorch.likelihoods.BernoulliLikelihood()

    model.train(); lik.train()
    opt = torch.optim.Adam([{'params': feat.parameters()},{'params': model.parameters()}], lr=cfg['lr'], weight_decay=cfg['wd'])
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=yT.numel())
    loader = DataLoader(TensorDataset(XT, yT), batch_size=batch_size, shuffle=True)

    best_auc, best_state, wait = -1.0, None, 0
    for epoch in range(max_epochs):
        for xb, yb in loader:
            opt.zero_grad(); fb = model(feat(xb)); loss = -mll(fb, yb); loss.backward(); opt.step()
        model.eval(); lik.eval()
        with torch.no_grad(): p_val = lik(model(feat(XV))).probs.cpu().numpy()
        val_auc = roc_auc_score(y_val, p_val)
        model.train(); lik.train()
        if val_auc > best_auc + 1e-4:
            best_auc, wait = val_auc, 0
            best_state = {
                'feat': {k: v.detach().cpu().clone() for k,v in feat.state_dict().items()},
                'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'scaler_mean': scaler.mean_.copy(), 'scaler_scale': scaler.scale_.copy()
            }
        else:
            wait += 1
            if wait >= patience: break

    feat.load_state_dict(best_state['feat']); model.load_state_dict(best_state['model'])
    scaler.mean_, scaler.scale_ = best_state['scaler_mean'], best_state['scaler_scale']
    X_full_t = torch.tensor(scaler.transform(X_full), dtype=torch.float32)

    model.eval(); lik.eval()
    with torch.no_grad():
        p_train_full = lik(model(feat(X_full_t))).probs.cpu().numpy()
        p_test       = lik(model(feat(XTe))).probs.cpu().numpy()
    return p_train_full, p_test

param_grid = [
    {'hidden':64,  'feat_dim':32, 'dropout':0.1, 'm':128, 'lr':1e-3, 'wd':1e-4},
    {'hidden':128, 'feat_dim':32, 'dropout':0.2, 'm':256, 'lr':1e-3, 'wd':1e-4},
    {'hidden':128, 'feat_dim':64, 'dropout':0.2, 'm':256, 'lr':5e-4,'wd':2e-4},
]

best_cfg, best_cv_auc = None, -1.0
for cfg in param_grid:
    auc = cv_mean_auc(X, y, cfg, n_splits=5, seed=2025)
    if auc > best_cv_auc:
        best_cv_auc, best_cfg = auc, cfg

p_train, p_test = fit_full_and_predict(X, y, X_test_all, best_cfg, max_epochs=60, batch_size=64, patience=10)

pd.DataFrame({'prob': p_train.ravel()}).to_csv('train_pred_gp.csv', index=False)
pd.DataFrame({'prob': p_test.ravel()}).to_csv('test_pred_gp.csv', index=False)
pd.DataFrame([{**best_cfg, 'mean_cv_auc': best_cv_auc}]).to_csv('best_hparams_gp.csv', index=False)
")

train_pred <- read.csv("train_pred_gp.csv")
test_pred  <- read.csv("test_pred_gp.csv")

train_probe <- data.frame(row.names = seq_len(nrow(dev)))
test_probe  <- data.frame(row.names = seq_len(nrow(vad)))
train_probe$GP_NeuralKernel <- train_pred$prob
test_probe$GP_NeuralKernel  <- test_pred$prob
