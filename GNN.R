library(torch)
library(caret)
library(pROC)

set.seed(XXXX)

train <- dev
test  <- vad[, var]
train$Result <- ifelse(train$Result == "Yes", 1L, 0L)
test$Result  <- ifelse(test$Result  == "Yes", 1L, 0L)

X_all_tr <- as.matrix(train[, setdiff(names(train), "Result"), drop = FALSE])
y_all_tr <- as.integer(train$Result)
X_te_raw <- as.matrix(test[,  setdiff(names(test),  "Result"), drop = FALSE])
y_te     <- as.integer(test$Result)

scale_fit   <- function(X){ mu <- colMeans(X); sdv <- apply(X,2,sd); sdv[sdv==0] <- 1; list(mu=mu, sd=sdv) }
scale_apply <- function(X,s){ (X - matrix(s$mu, nrow(X), byrow=TRUE)) / matrix(s$sd, nrow(X), byrow=TRUE) }

norm_adj <- function(X){             # 余弦相似度+ReLU+self-loop+对称归一化
  Xn <- X / sqrt(rowSums(X^2)+1e-8)
  S  <- Xn %*% t(Xn); S[S<0] <- 0; diag(S) <- 1
  D  <- diag(rowSums(S)+1e-8)
  Ahat <- solve(sqrt(D)) %*% S %*% solve(sqrt(D))
  torch_tensor(Ahat, dtype=torch_float())
}

gcn <- nn_module(
  initialize = function(in_dim, hidden, out_dim=2, dropout=0.3){
    self$lin1 <- nn_linear(in_dim, hidden)
    self$lin2 <- nn_linear(hidden, out_dim)
    self$drop <- dropout
  },
  forward = function(X, A){
    H <- self$lin1(X)
    H <- torch_mm(A, H)
    H <- torch_relu(H)
    H <- nnf_dropout(H, p=self$drop, training=self$training)
    Z <- self$lin2(H)
    Z <- torch_mm(A, Z)
    Z
  }
)

train_one_single_graph <- function(X_all, y_all, tr_idx, va_idx,
                                   hidden=64, dropout=0.3, lr=1e-3, l2=3e-4,
                                   epochs=100, patience=10){
  sfit   <- scale_fit(X_all[tr_idx,,drop=FALSE])
  Xs_all <- scale_apply(X_all, sfit)
  A_full <- norm_adj(Xs_all)
  
  X_t <- torch_tensor(Xs_all, dtype=torch_float())
  y_t <- torch_tensor(y_all,  dtype=torch_long())
  A_t <- A_full
  
  tr_idx_t <- torch_tensor(as.integer(tr_idx-1L), dtype=torch_long())  # 0-based
  va_idx_t <- torch_tensor(as.integer(va_idx-1L), dtype=torch_long())
  
  model <- gcn(ncol(X_all), hidden=hidden, dropout=dropout)
  opt   <- optim_adam(model$parameters, lr=lr, weight_decay=l2)
  crit  <- nn_cross_entropy_loss()
  
  best_auc <- -Inf; best_state <- NULL; wait <- 0L
  
  for(ep in 1:epochs){
    model$train()
    opt$zero_grad()
    logits_all <- model(X_t, A_t)
    logits_tr  <- logits_all$index_select(1, tr_idx_t)
    y_tr_t     <- y_t$index_select(1, tr_idx_t)
    loss <- crit(logits_tr, y_tr_t)
    loss$backward(); opt$step()
    
    model$eval()
    with_no_grad({
      logits_all <- model(X_t, A_t)
      logits_va  <- logits_all$index_select(1, va_idx_t)
      prob_va    <- as_array(nnf_softmax(logits_va, dim=2)[,2])
      auc_va     <- as.numeric(pROC::auc(pROC::roc(y_all[va_idx], prob_va, quiet=TRUE)))
    })
    if (!is.finite(auc_va)) auc_va <- -Inf
    if (auc_va > best_auc + 1e-4){
      best_auc <- auc_va; wait <- 0L; best_state <- model$state_dict()
    } else {
      wait <- wait + 1L
      if (wait >= patience) break
    }
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  list(model=model, scaler=sfit, A_full=A_full, val_auc=best_auc)
}

predict_prob_single_graph <- function(model, X_all, scaler, A_full){
  Xs_all <- scale_apply(X_all, scaler)
  X_t <- torch_tensor(Xs_all, dtype=torch_float())
  model$eval()
  with_no_grad({
    logits <- model(X_t, A_full)
    as_array(nnf_softmax(logits, dim=2)[,2])
  })
}

folds <- createFolds(y_all_tr, k=5, list=TRUE, returnTrain=FALSE)

grid <- expand.grid(
  hidden  = c(64L, 128L),
  dropout = c(0.2, 0.3),
  l2      = c(1e-4, 3e-4),
  lr      = c(1e-3, 5e-4),
  stringsAsFactors = FALSE
)
grid$mean_auc <- NA_real_

for(i in seq_len(nrow(grid))){
  gi <- grid[i,]
  aucs <- numeric(length(folds))
  
  for(k in seq_along(folds)){
    va_idx <- folds[[k]]
    tr_idx <- setdiff(seq_len(nrow(X_all_tr)), va_idx)
    
    fitk <- train_one_single_graph(
      X_all = X_all_tr, y_all = y_all_tr,
      tr_idx = tr_idx, va_idx = va_idx,
      hidden = gi$hidden, dropout = gi$dropout, lr = gi$lr, l2 = gi$l2,
      epochs = 100, patience = 10
    )
    aucs[k] <- fitk$val_auc
    torch_cuda_empty_cache()
  }
  grid$mean_auc[i] <- mean(aucs, na.rm=TRUE)
}

grid <- grid[order(-grid$mean_auc, grid$hidden, grid$dropout, grid$l2, grid$lr),]
best <- grid[1,,drop=FALSE]
print(grid)
cat(sprintf("Best -> hidden=%d, dropout=%.2f, L2=%.4g, lr=%.4g | mean CV AUC=%.4f\n",
            best$hidden, best$dropout, best$l2, best$lr, best$mean_auc))

hold <- createDataPartition(y_all_tr, p=0.85, list=FALSE)
tr_idx_full <- as.vector(hold)
va_idx_full <- setdiff(seq_len(nrow(X_all_tr)), tr_idx_full)

final_fit <- train_one_single_graph(
  X_all = X_all_tr, y_all = y_all_tr,
  tr_idx = tr_idx_full, va_idx = va_idx_full,
  hidden = best$hidden, dropout = best$dropout, lr = best$lr, l2 = best$l2,
  epochs = 100, patience = 10
)
final_model  <- final_fit$model
final_scaler <- final_fit$scaler
A_full_tr    <- final_fit$A_full

prob_tr_all <- predict_prob_single_graph(final_model, X_all_tr, final_scaler, A_full_tr)
auc_tr <- as.numeric(pROC::auc(pROC::roc(y_all_tr, prob_tr_all, quiet=TRUE)))
cat(sprintf("Final Train AUC: %.4f\n", auc_tr))

sfit_te <- final_scaler
Xs_te   <- scale_apply(X_te_raw, sfit_te)
A_te    <- norm_adj(Xs_te)
Xte_t   <- torch_tensor(Xs_te, dtype=torch_float())
final_model$eval()
with_no_grad({
  logits_te <- final_model(Xte_t, A_te)
  prob_te   <- as_array(nnf_softmax(logits_te, dim=2)[,2])
})
auc_te <- as.numeric(pROC::auc(pROC::roc(y_te, prob_te, quiet=TRUE)))
cat(sprintf("Final Test  AUC: %.4f\n", auc_te))

train_probe <- data.frame(
  ID  = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
  GNN = prob_tr_all
)
test_probe <- data.frame(
  ID  = if ("ID" %in% names(test))  as.character(test$ID)  else sprintf("test_%d",  seq_len(nrow(test))),
  GNN = prob_te
)