suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(pROC)
  library(recipes)
  library(glmnet)
  library(gbm)
  library(kernlab)
  library(kknn)
  library(ada)
  library(xgboost)
  library(lightgbm)
  library(catboost)
})

train <- dev
test  <- if (exists("var")) vad[, c("Result", intersect(var, names(vad))), drop = FALSE] else vad
stopifnot("Result" %in% names(train), "Result" %in% names(test))

train$Result <- factor(train$Result, levels = c("Yes","No"))
test$Result  <- factor(test$Result,  levels = c("Yes","No"))
xvars <- setdiff(intersect(names(train), names(test)), "Result")

set.seed(XXXX)
folds <- caret::createFolds(train$Result, k = 5, returnTrain = TRUE)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = FALSE,
  allowParallel = TRUE
)

base_rec <- recipe(Result ~ ., data = cbind(Result = train$Result, train[xvars])) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_zv(all_predictors()) |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors())

rec_prep  <- prep(base_rec, training = cbind(Result=train$Result, train[xvars]))
Xtr_full  <- bake(rec_prep, new_data = cbind(Result=train$Result, train[xvars])) |> dplyr::select(-Result)
ytr_full  <- train$Result
pos_label <- as.integer(ytr_full == "Yes")
Xte_full  <- bake(rec_prep, new_data = cbind(Result=test$Result, test[xvars])) |> dplyr::select(-Result)

grid_glmnet <- expand.grid(alpha = c(0, 0.5, 1), lambda = 10^seq(-4, 1, length.out = 10))
grid_rf     <- expand.grid(mtry = pmax(1, round(c(sqrt(length(xvars)), length(xvars)/4, length(xvars)/2))))
grid_knn    <- expand.grid(k = c(3,5,7,9,11,15,21))
grid_gbm    <- expand.grid(
  n.trees = c(200, 400, 800),
  interaction.depth = c(1,2,3),
  shrinkage = c(0.05, 0.1),
  n.minobsinnode = c(10,20)
)
grid_ada    <- expand.grid(iter = c(100,200,400), maxdepth = c(1,2,3), nu = c(0.05,0.1))
set.seed(2025)
grid_xgb    <- expand.grid(
  nrounds = c(300, 600),
  max_depth = c(3,5,7),
  eta = c(0.05, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 5),
  subsample = c(0.7, 1.0)
)

set.seed(XXXX)
fit_glmnet <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_glmnet
)

set.seed(XXXX)
fit_rf <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_rf,
  ntree = 500
)

set.seed(XXXX)
fit_svm <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)

set.seed(XXXX)
fit_knn <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "knn",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_knn
)

set.seed(XXXX)
fit_gbm <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "gbm",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_gbm,
  verbose = FALSE
)

set.seed(XXXX)
fit_ada <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "ada",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_ada
)

set.seed(XXXX)
fit_xgb <- train(
  base_rec, data = cbind(Result=train$Result, train[xvars]),
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_xgb
)


lgb_grid <- expand.grid(
  num_leaves = c(15, 31, 63),
  learning_rate = c(0.05, 0.1),
  feature_fraction = c(0.8, 1.0),
  bagging_fraction = c(0.8, 1.0),
  bagging_freq = c(0, 1)
)

cat_grid <- expand.grid(
  depth = c(4, 6, 8),
  learning_rate = c(0.05, 0.1),
  l2_leaf_reg = c(3, 5, 7),
  border_count = c(64, 128)
)

eval_lgb_grid <- function(row){
  params <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = row$num_leaves,
    learning_rate = row$learning_rate,
    feature_fraction = row$feature_fraction,
    bagging_fraction = row$bagging_fraction,
    bagging_freq = row$bagging_freq,
    is_unbalance = TRUE,
    verbose = -1
  )
  oof <- rep(NA_real_, nrow(Xtr_full))
  best_iters <- integer(length(folds))
  for(i in seq_along(folds)){
    tr_idx <- folds[[i]]
    va_idx <- setdiff(seq_len(nrow(Xtr_full)), tr_idx)
    dtr <- lgb.Dataset(as.matrix(Xtr_full[tr_idx, , drop=FALSE]), label = pos_label[tr_idx])
    dva <- lgb.Dataset(as.matrix(Xtr_full[va_idx, , drop=FALSE]),  label = pos_label[va_idx])
    m <- lgb.train(
      params = params,
      data = dtr,
      nrounds = 2000,
      valids = list(val = dva),
      early_stopping_rounds = 100,
      verbose = -1
    )
    oof[va_idx] <- predict(m, as.matrix(Xtr_full[va_idx, , drop=FALSE]))
    best_iters[i] <- m$best_iter
  }
  auc <- pROC::auc(pROC::roc(ytr_full, oof, levels = c("No","Yes"), direction = "<", quiet = TRUE))
  list(mean_auc = as.numeric(auc), mean_best_iter = round(mean(best_iters)))
}

eval_cat_grid <- function(row){
  params <- list(
    loss_function = "Logloss",
    eval_metric = "AUC",
    depth = row$depth,
    learning_rate = row$learning_rate,
    l2_leaf_reg = row$l2_leaf_reg,
    border_count = row$border_count,
    od_type = "Iter",
    od_wait = 100,
    random_seed = 2025,
    verbose = FALSE,
    auto_class_weights = "Balanced"
  )
  oof <- rep(NA_real_, nrow(Xtr_full))
  best_iters <- integer(length(folds))
  for(i in seq_along(folds)){
    tr_idx <- folds[[i]]
    va_idx <- setdiff(seq_len(nrow(Xtr_full)), tr_idx)
    pool_tr <- catboost.load_pool(as.matrix(Xtr_full[tr_idx, , drop=FALSE]), label = pos_label[tr_idx])
    pool_va <- catboost.load_pool(as.matrix(Xtr_full[va_idx, , drop=FALSE]),  label = pos_label[va_idx])
    m <- catboost.train(pool = pool_tr, params = params)
    oof[va_idx] <- as.numeric(catboost.predict(m, pool_va, prediction_type = "Probability"))
    best_iters[i] <- m$tree_count
  }
  auc <- pROC::auc(pROC::roc(ytr_full, oof, levels = c("No","Yes"), direction = "<", quiet = TRUE))
  list(mean_auc = as.numeric(auc), mean_best_iter = round(mean(best_iters)))
}

lgb_tune <- do.call(
  rbind,
  lapply(seq_len(nrow(lgb_grid)), function(i){
    res <- eval_lgb_grid(lgb_grid[i, ])
    cbind(lgb_grid[i, ], data.frame(mean_auc = res$mean_auc, best_iter = res$mean_best_iter))
  })
)
lgb_tune <- lgb_tune[order(-lgb_tune$mean_auc), ]
best_lgb <- lgb_tune[1, ]

cat_tune <- do.call(
  rbind,
  lapply(seq_len(nrow(cat_grid)), function(i){
    res <- eval_cat_grid(cat_grid[i, ])
    cbind(cat_grid[i, ], data.frame(mean_auc = res$mean_auc, best_iter = res$mean_best_iter))
  })
)
cat_tune <- cat_tune[order(-cat_tune$mean_auc), ]
best_cat <- cat_tune[1, ]

lgb_oof <- rep(NA_real_, nrow(train))
cat_oof <- rep(NA_real_, nrow(train))

params_lgb_best <- list(
  objective = "binary",
  metric = "auc",
  num_leaves = best_lgb$num_leaves,
  learning_rate = best_lgb$learning_rate,
  feature_fraction = best_lgb$feature_fraction,
  bagging_fraction = best_lgb$bagging_fraction,
  bagging_freq = best_lgb$bagging_freq,
  is_unbalance = TRUE,
  verbose = -1
)

params_cat_best <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  depth = best_cat$depth,
  learning_rate = best_cat$learning_rate,
  l2_leaf_reg = best_cat$l2_leaf_reg,
  border_count = best_cat$border_count,
  od_type = "Iter",
  od_wait = 100,
  random_seed = 2025,
  verbose = FALSE,
  auto_class_weights = "Balanced"
)

for(i in seq_along(folds)){
  tr_idx <- folds[[i]]
  va_idx <- setdiff(seq_len(nrow(train)), tr_idx)
  
  dtr_lgb <- lgb.Dataset(as.matrix(Xtr_full[tr_idx, , drop = FALSE]), label = pos_label[tr_idx])
  dval_lgb <- lgb.Dataset(as.matrix(Xtr_full[va_idx, , drop = FALSE]), label = pos_label[va_idx])
  m_lgb <- lgb.train(
    params = params_lgb_best,
    data = dtr_lgb,
    nrounds = 2000,
    valids = list(val = dval_lgb),
    early_stopping_rounds = 100,
    verbose = -1
  )
  lgb_oof[va_idx] <- predict(m_lgb, as.matrix(Xtr_full[va_idx, , drop = FALSE]))
  
  pool_tr <- catboost.load_pool(as.matrix(Xtr_full[tr_idx, , drop = FALSE]), label = pos_label[tr_idx])
  pool_va <- catboost.load_pool(as.matrix(Xtr_full[va_idx, , drop = FALSE]))
  m_cat <- catboost.train(pool = pool_tr, params = params_cat_best)
  cat_oof[va_idx] <- as.numeric(catboost.predict(m_cat, pool_va, prediction_type = "Probability"))
}

get_oof <- function(fit){
  bt <- fit$bestTune
  pr <- merge(fit$pred, bt, by = intersect(names(fit$pred), names(bt)))
  pr <- pr[order(pr$rowIndex), c("rowIndex", "Yes")]
  pr
}

oof_glmnet <- get_oof(fit_glmnet)
oof_rf     <- get_oof(fit_rf)
oof_svm    <- get_oof(fit_svm)
oof_knn    <- get_oof(fit_knn)
oof_gbm    <- get_oof(fit_gbm)
oof_ada    <- get_oof(fit_ada)
oof_xgb    <- get_oof(fit_xgb)

train_probe <- data.frame(
  ID    = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
  GLMNET = oof_glmnet$Yes,
  RF     = oof_rf$Yes,
  SVM    = oof_svm$Yes,
  KNN    = oof_knn$Yes,
  GBM    = oof_gbm$Yes,
  ADA    = oof_ada$Yes,
  XGB    = oof_xgb$Yes,
  LGB    = lgb_oof,
  CAT    = cat_oof,
  stringsAsFactors = FALSE
)

pred_glmnet_te <- predict(fit_glmnet, newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_rf_te     <- predict(fit_rf,     newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_svm_te    <- predict(fit_svm,    newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_knn_te    <- predict(fit_knn,    newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_gbm_te    <- predict(fit_gbm,    newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_ada_te    <- predict(fit_ada,    newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]
pred_xgb_te    <- predict(fit_xgb,    newdata = cbind(Result=test$Result, test[xvars]), type = "prob")[, "Yes"]

dtrain_lgb_full <- lgb.Dataset(as.matrix(Xtr_full), label = pos_label)
m_lgb_full <- lgb.train(params = params_lgb_best, data = dtrain_lgb_full, nrounds = best_lgb$best_iter, verbose = -1)
pred_lgb_te <- predict(m_lgb_full, as.matrix(Xte_full))

pool_tr_full <- catboost.load_pool(as.matrix(Xtr_full), label = pos_label)
m_cat_full <- catboost.train(pool = pool_tr_full, params = params_cat_best)
pred_cat_te <- as.numeric(catboost.predict(m_cat_full, catboost.load_pool(as.matrix(Xte_full)), prediction_type = "Probability"))

test_probe <- data.frame(
  ID    = if ("ID" %in% names(test)) as.character(test$ID) else sprintf("test_%d", seq_len(nrow(test))),
  GLMNET = pred_glmnet_te,
  RF     = pred_rf_te,
  SVM    = pred_svm_te,
  KNN    = pred_knn_te,
  GBM    = pred_gbm_te,
  ADA    = pred_ada_te,
  XGB    = pred_xgb_te,
  LGB    = pred_lgb_te,
  CAT    = pred_cat_te,
  stringsAsFactors = FALSE
)
