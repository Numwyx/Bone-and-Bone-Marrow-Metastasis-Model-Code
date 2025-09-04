library(reticulate)
library(caret)
library(pROC)
library(precrec)

use_condaenv("pyws", required = TRUE)
tabnet <- import("pytorch_tabnet.tab_model")
np     <- import("numpy")
torch  <- import("torch")
optim  <- import("torch.optim")

train_df <- dev
test_df  <- vad[, var]
train_df$Result <- as.integer(train_df$Result == "Yes")
test_df$Result  <- as.integer(test_df$Result  == "Yes")

fit_preprocess <- function(df) {
  y <- df$Result
  X <- df[, setdiff(names(df), "Result"), drop = FALSE]
  dv <- caret::dummyVars(~ ., data = X, fullRank = TRUE)
  Xm <- predict(dv, newdata = X)
  pp <- caret::preProcess(Xm, method = c("center", "scale"))
  list(dv = dv, pp = pp)
}
apply_preprocess <- function(df, dv, pp) {
  X <- df[, setdiff(names(df), "Result"), drop = FALSE]
  Xm <- predict(dv, newdata = X)
  Xt <- predict(pp, Xm)
  Xt
}

folds <- caret::createFolds(train_df$Result, k = 5, list = TRUE, returnTrain = FALSE)

grid <- expand.grid(
  dropout = c(0.2, 0.3),
  l2      = c(1e-4, 3e-4),     # 作为 Adam 的 weight_decay
  lr      = c(1e-2, 5e-3, 1e-3),
  stringsAsFactors = FALSE
)

train_eval_fold <- function(tr_idx, va_idx, params) {
  tr <- train_df[tr_idx, , drop = FALSE]
  va <- train_df[va_idx, , drop = FALSE]
  
  pp_fit <- fit_preprocess(tr)
  X_tr <- apply_preprocess(tr, pp_fit$dv, pp_fit$pp)
  X_va <- apply_preprocess(va, pp_fit$dv, pp_fit$pp)
  
  X_tr_np <- np$array(as.matrix(X_tr), dtype = "float32")
  y_tr_np <- np$array(tr$Result, dtype = "int64")
  X_va_np <- np$array(as.matrix(X_va), dtype = "float32")
  y_va_np <- np$array(va$Result, dtype = "int64")
  
  y_r <- tr$Result
  tbl <- table(y_r)
  w_pos <- as.numeric(1 / tbl["1"])
  w_neg <- as.numeric(1 / tbl["0"])
  sw    <- ifelse(y_r == 1L, w_pos, w_neg)
  sw_np <- np$array(sw, dtype = "float32")
  
  model <- tabnet$TabNetClassifier(
    n_d = as.integer(24), n_a = as.integer(24),
    n_steps = as.integer(3), gamma = 1.3,
    lambda_sparse = 1e-4,
    optimizer_fn = optim$Adam,
    optimizer_params = dict(lr = params$lr, weight_decay = params$l2),
    mask_type = "sparsemax",
    scheduler_params = dict(step_size = as.integer(15), gamma = 0.9),
    scheduler_fn = import("torch.optim.lr_scheduler")$StepLR,
    dropout = params$dropout,
    seed = as.integer(2025)
  )
  
  model$fit(
    X_train = X_tr_np,
    y_train = y_tr_np,
    eval_set = list(list(X_va_np, y_va_np)),
    eval_name = list("valid"),
    eval_metric = list("auc"),
    max_epochs = as.integer(100),
    patience = as.integer(20),
    batch_size = as.integer(256),
    virtual_batch_size = as.integer(64),
    num_workers = as.integer(0),
    drop_last = FALSE,
    weights = sw_np,
    verbose = 0L
  )
  
  prob_va <- py_to_r(model$predict_proba(X_va_np))[, 2]
  as.numeric(pROC::auc(pROC::roc(va$Result, prob_va, quiet = TRUE)))
}

grid$mean_auc <- NA_real_
for (i in seq_len(nrow(grid))) {
  params <- grid[i, ]
  aucs <- numeric(length(folds))
  for (k in seq_along(folds)) {
    va_idx <- folds[[k]]
    tr_idx <- setdiff(seq_len(nrow(train_df)), va_idx)
    aucs[k] <- train_eval_fold(tr_idx, va_idx, params)
  }
  grid$mean_auc[i] <- mean(aucs, na.rm = TRUE)
}

grid <- grid[order(-grid$mean_auc, grid$dropout, grid$l2, grid$lr), ]
best  <- grid[1, , drop = FALSE]

print(grid)
cat(sprintf("Best params -> dropout=%.2f, l2=%.4g, lr=%.4g | mean CV AUC=%.4f\n",
            best$dropout, best$l2, best$lr, best$mean_auc))

pp_full <- fit_preprocess(train_df)
X_train_full <- apply_preprocess(train_df, pp_full$dv, pp_full$pp)
X_test_full  <- apply_preprocess(test_df,  pp_full$dv, pp_full$pp)

X_tr_np <- np$array(as.matrix(X_train_full), dtype = "float32")
y_tr_np <- np$array(train_df$Result, dtype = "int64")
X_te_np <- np$array(as.matrix(X_test_full), dtype = "float32")
y_te_np <- np$array(test_df$Result, dtype = "int64")

y_r <- train_df$Result
tbl <- table(y_r)
w_pos <- as.numeric(1 / tbl["1"])
w_neg <- as.numeric(1 / tbl["0"])
sw    <- ifelse(y_r == 1L, w_pos, w_neg)
sw_np <- np$array(sw, dtype = "float32")

final_model <- tabnet$TabNetClassifier(
  n_d = as.integer(24), n_a = as.integer(24),
  n_steps = as.integer(3), gamma = 1.3,
  lambda_sparse = 1e-4,
  optimizer_fn = optim$Adam,
  optimizer_params = dict(lr = best$lr, weight_decay = best$l2),
  mask_type = "sparsemax",
  scheduler_params = dict(step_size = as.integer(15), gamma = 0.9),
  scheduler_fn = import("torch.optim.lr_scheduler")$StepLR,
  dropout = best$dropout,
  seed = as.integer(2025)
)

final_model$fit(
  X_train = X_tr_np,
  y_train = y_tr_np,
  eval_set = list(list(X_te_np, y_te_np)),
  eval_name = list("valid"),
  eval_metric = list("auc"),
  max_epochs = as.integer(100),
  patience = as.integer(20),
  batch_size = as.integer(256),
  virtual_batch_size = as.integer(64),
  num_workers = as.integer(0),
  drop_last = FALSE,
  weights = sw_np,
  verbose = 0L
)

train_prob <- py_to_r(final_model$predict_proba(X_tr_np))[, 2]
test_prob  <- py_to_r(final_model$predict_proba(X_te_np))[, 2]