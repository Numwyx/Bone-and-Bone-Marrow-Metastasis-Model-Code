library(reticulate)
library(caret)
library(pROC)

use_condaenv("pyws", required = TRUE)
np <- import("numpy")
SVC <- import("sklearn.svm")$SVC
import("pennylane")

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
import pennylane as qml
from sklearn.svm import SVC

def make_qkernel(n_qubits:int, reps:int=1, scale:float=1.0):
    dev = qml.device('default.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def overlap(x1, x2):
        for _ in range(reps):
            qml.templates.AngleEmbedding(scale*x1, wires=range(n_qubits))
            qml.adjoint(qml.templates.AngleEmbedding)(scale*x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))
    def kernel(X1, X2):
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                p = overlap(X1[i], X2[j])
                K[i, j] = np.sum(np.sqrt(p))
        return K
    return kernel

def fit_svc_precomputed(K_train, y_train, C=1.0, seed=2025):
    clf = SVC(kernel='precomputed', probability=True, C=C, random_state=seed)
    clf.fit(K_train, y_train)
    return clf

def predict_proba_svc(clf, K):
    return clf.predict_proba(K)[:,1]
")

folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = FALSE)

grid <- expand.grid(
  reps  = c(1L, 2L, 3L),
  scale = c(0.1, 0.5, 1.0, 2.0),
  C     = c(0.01, 0.1, 1.0, 10.0, 100.0),
  stringsAsFactors = FALSE
)
grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))) {
  reps_i  <- grid$reps[i]
  scale_i <- grid$scale[i]
  C_i     <- grid$C[i]
  
  aucs <- numeric(length(folds))
  for (k in seq_along(folds)) {
    va_idx <- folds[[k]]
    tr_idx <- setdiff(seq_len(nrow(x_train)), va_idx)
    
    Xtr <- np$array(x_train[tr_idx, , drop = FALSE], dtype = "float64")
    Xva <- np$array(x_train[va_idx, , drop = FALSE],  dtype = "float64")
    ytr <- np$array(y_train[tr_idx], dtype = "int64")
    yva <- y_train[va_idx]
    
    kernel_fun <- py$make_qkernel(as.integer(ncol(x_train)), as.integer(reps_i), as.numeric(scale_i))
    K_tr <- kernel_fun(Xtr, Xtr)
    K_va <- kernel_fun(Xva, Xtr)
    
    clf  <- py$fit_svc_precomputed(K_tr, ytr, C = as.numeric(C_i))
    p_va <- as.numeric(py$predict_proba_svc(clf, K_va))
    
    aucs[k] <- as.numeric(pROC::auc(pROC::roc(yva, p_va, quiet = TRUE)))
  }
  
  grid$mean_auc[i] <- mean(aucs, na.rm = TRUE)
}

grid <- grid[order(-grid$mean_auc, grid$reps, grid$scale, grid$C), ]
print(grid[ , c("reps","scale","C","mean_auc")])
best <- grid[1, ]
cat(sprintf("Best params -> reps=%d, scale=%.2f, C=%.3f | mean CV AUC=%.4f\n",
            best$reps, best$scale, best$C, best$mean_auc))

kernel_fun_full <- py$make_qkernel(as.integer(ncol(x_train)), as.integer(best$reps), as.numeric(best$scale))
Xtr_full <- np$array(x_train, dtype = "float64")
Xte_full <- np$array(x_test,  dtype = "float64")
ytr_full <- np$array(y_train, dtype = "int64")

K_train_full <- kernel_fun_full(Xtr_full, Xtr_full)
K_test_full  <- kernel_fun_full(Xte_full,  Xtr_full)

clf_full   <- py$fit_svc_precomputed(K_train_full, ytr_full, C = as.numeric(best$C))
train_prob <- as.numeric(py$predict_proba_svc(clf_full, K_train_full))
test_prob  <- as.numeric(py$predict_proba_svc(clf_full, K_test_full))

train_probe <- data.frame(
  ID  = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
  QML = train_prob
)
test_probe <- data.frame(
  ID  = if ("ID" %in% names(test))  as.character(test$ID)  else sprintf("test_%d",  seq_len(nrow(test))),
  QML = test_prob
)
