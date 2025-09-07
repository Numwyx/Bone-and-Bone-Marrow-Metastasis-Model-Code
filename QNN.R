library(reticulate)
library(caret)
library(pROC)

use_condaenv("pyws", required = TRUE)
np <- import("numpy")

train <- dev
test  <- vad[, var]
train$Result <- ifelse(train$Result == "Yes", 1L, 0L)
test$Result  <- ifelse(test$Result  == "Yes", 1L, 0L)

x_train <- as.matrix(train[, setdiff(names(train), "Result"), drop = FALSE])
y_train <- as.integer(train$Result)
x_test  <- as.matrix(test[,  setdiff(names(test),  "Result"), drop = FALSE])
y_test  <- as.integer(test$Result)

py$x_dim <- as.integer(ncol(x_train))

py_run_string("
import numpy as np
import torch
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float32)

class VQC(torch.nn.Module):
    def __init__(self, n_qubits:int, reps:int=1, n_layers:int=2, scale:float=1.0, seed:int=2025):
        super().__init__()
        self.n_qubits  = n_qubits
        self.reps      = reps
        self.n_layers  = n_layers
        self.scale     = scale
        self.dev       = qml.device('default.qubit', wires=self.n_qubits, shots=None)
        torch.manual_seed(seed)
        self.weights   = torch.nn.Parameter(0.01*torch.randn(self.n_layers, self.n_qubits, 3))

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(x, weights):
            for _ in range(self.reps):
                qml.templates.AngleEmbedding(self.scale * x, wires=range(self.n_qubits))
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward_one(self, x):
        z = self.circuit(x, self.weights)
        p = (1.0 + z) * 0.5
        return p

    def forward_batch(self, X):
        probs = [self.forward_one(x) for x in X]
        return torch.stack(probs).reshape(-1, 1)

def train_eval_vqc(X_tr, y_tr, X_va, y_va, *,
                   n_qubits:int, reps:int, n_layers:int, scale:float,
                   lr:float=1e-2, epochs:int=60, batch_size:int=64, patience:int=10, seed:int=2025):
    scaler = StandardScaler().fit(X_tr)
    XT = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    XV = torch.tensor(scaler.transform(X_va), dtype=torch.float32)
    yT = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)
    yV = torch.tensor(y_va.reshape(-1,1), dtype=torch.float32)

    model = VQC(n_qubits=n_qubits, reps=reps, n_layers=n_layers, scale=scale, seed=seed)
    opt   = torch.optim.Adam([{'params': model.parameters(), 'lr': lr}])
    bce   = torch.nn.BCELoss()
    loader = DataLoader(TensorDataset(XT, yT), batch_size=batch_size, shuffle=True)

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pb = model.forward_batch(xb).clamp(1e-6, 1 - 1e-6)
            loss = bce(pb, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model.forward_batch(XV).cpu().numpy().ravel()
            auc = roc_auc_score(y_va, pv)
        if auc > best_auc + 1e-4:
            best_auc = auc; wait = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_scaler = (scaler.mean_.copy(), scaler.scale_.copy())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    scaler.mean_, scaler.scale_ = best_scaler
    return float(best_auc), model.state_dict(), (scaler.mean_.copy(), scaler.scale_.copy())

def fit_full_and_predict_vqc(X_full, y_full, X_test, *,
                             n_qubits:int, reps:int, n_layers:int, scale:float,
                             lr:float=1e-2, epochs:int=60, batch_size:int=64, patience:int=10, val_size:float=0.15, seed:int=2025):
    rs = np.random.RandomState(seed)
    idx = np.arange(X_full.shape[0]); rs.shuffle(idx)
    n_val = int(round(val_size * len(idx)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]

    auc, state, scaler_stats = train_eval_vqc(
        X_full[tr_idx], y_full[tr_idx], X_full[va_idx], y_full[va_idx],
        n_qubits=n_qubits, reps=reps, n_layers=n_layers, scale=scale,
        lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, seed=seed
    )

    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = scaler_stats
    XT  = torch.tensor(scaler.transform(X_full), dtype=torch.float32)
    XTe = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    model = VQC(n_qubits=n_qubits, reps=reps, n_layers=n_layers, scale=scale, seed=seed)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        p_train = model.forward_batch(XT).cpu().numpy().ravel()
        p_test  = model.forward_batch(XTe).cpu().numpy().ravel()
    return p_train, p_test, auc
")

folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = FALSE)

grid <- expand.grid(
  reps      = c(1L, 2L),
  n_layers  = c(1L, 2L, 3L),
  scale     = c(0.5, 1.0),
  lr        = c(1e-2, 5e-3, 1e-3),
  batch_sz  = c(32L, 64L),
  stringsAsFactors = FALSE
)
grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))) {
  gi <- grid[i, ]
  aucs <- numeric(length(folds))
  for (k in seq_along(folds)) {
    va_idx <- folds[[k]]
    tr_idx <- setdiff(seq_len(nrow(x_train)), va_idx)
    
    Xtr <- np$array(x_train[tr_idx, , drop = FALSE], dtype = "float64")
    Xva <- np$array(x_train[va_idx, , drop = FALSE],  dtype = "float64")
    ytr <- np$array(y_train[tr_idx], dtype = "int64")
    yva <- y_train[va_idx]
    
    out <- py$train_eval_vqc(
      X_tr = Xtr, y_tr = ytr, X_va = Xva, y_va = yva,
      n_qubits = as.integer(ncol(x_train)),
      reps = as.integer(gi$reps),
      n_layers = as.integer(gi$n_layers),
      scale = as.numeric(gi$scale),
      lr = as.numeric(gi$lr),
      epochs = as.integer(60),
      batch_size = as.integer(gi$batch_sz),
      patience = as.integer(10),
      seed = as.integer(xxxx)
    )
    aucs[k] <- as.numeric(out[[1]])
  }
  grid$mean_auc[i] <- mean(aucs, na.rm = TRUE)
}

grid <- grid[order(-grid$mean_auc, grid$n_layers, grid$reps, grid$scale, grid$lr, grid$batch_sz), ]
print(grid[, c('reps','n_layers','scale','lr','batch_sz','mean_auc')])

best <- grid[1, ]
cat(sprintf(
  'Best params -> reps=%d, n_layers=%d, scale=%.2f, lr=%.4g, batch=%d | mean CV AUC=%.4f\n',
  best$reps, best$n_layers, best$scale, best$lr, best$batch_sz, best$mean_auc
))

Xtr_full <- np$array(x_train, dtype = "float64")
Xte_full <- np$array(x_test,  dtype = "float64")
ytr_full <- np$array(y_train, dtype = "int64")

fit_out <- py$fit_full_and_predict_vqc(
  X_full = Xtr_full, y_full = ytr_full, X_test = Xte_full,
  n_qubits = as.integer(ncol(x_train)),
  reps = as.integer(best$reps),
  n_layers = as.integer(best$n_layers),
  scale = as.numeric(best$scale),
  lr = as.numeric(best$lr),
  epochs = as.integer(60),
  batch_size = as.integer(best$batch_sz),
  patience = as.integer(10),
  val_size = as.numeric(0.15),
  seed = as.integer(xxxx)
)

train_prob <- as.numeric(fit_out[[1]])
test_prob  <- as.numeric(fit_out[[2]])
cat(sprintf('Internal holdout AUC (for monitoring): %.4f\n', as.numeric(fit_out[[3]])))

train_probe <- data.frame(
  ID  = if ('ID' %in% names(train)) as.character(train$ID) else sprintf('train_%d', seq_len(nrow(train))),
  QNN = train_prob
)
test_probe <- data.frame(
  ID  = if ('ID' %in% names(test))  as.character(test$ID)  else sprintf('test_%d',  seq_len(nrow(test))),
  QNN = test_prob
)
