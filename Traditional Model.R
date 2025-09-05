suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(pROC)
  library(Matrix)
  library(glmnet)
  library(gbm)
  library(xgboost)
  library(randomForest)
  library(e1071)
  library(kknn)
  library(ada)         # AdaBoost (caret method = "ada")
  library(lightgbm)    # LightGBM native R
  library(catboost)    # CatBoost native R
})

train <- dev
test  <- if (exists("var")) vad[, c("Result", intersect(var, names(vad))), drop = FALSE] else vad
stopifnot("Result" %in% names(train), "Result" %in% names(test))

train$Result <- factor(train$Result, levels = c("Yes","No"))
test$Result  <- factor(test$Result,  levels = c("Yes","No"))

xvars <- setdiff(intersect(names(train), names(test)), "Result")

dv <- caret::dummyVars(~ ., data = train[xvars], fullRank = TRUE)

Xtr_mm <- predict(dv, newdata = train[xvars]) %>% as.data.frame()
Xte_mm <- predict(dv, newdata = test[xvars])  %>% as.data.frame()

pp <- preProcess(Xtr_mm, method = c("center","scale"))
Xtr <- predict(pp, Xtr_mm)
Xte <- predict(pp, Xte_mm)

ytr <- train$Result
yte <- test$Result

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = NULL,
  verboseIter = FALSE,
  allowParallel = TRUE
)

grid_glmnet <- expand.grid(alpha = c(0, 0.5, 1), lambda = 10^seq(-4, 1, length.out = 10))
fit_glmnet <- train(
  x = Xtr, y = ytr,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_glmnet,
  family = "binomial",
  standardize = FALSE
)
best_glmnet <- fit_glmnet$bestTune
auc_glmnet  <- max(fit_glmnet$results$ROC)

grid_rf <- expand.grid(mtry = pmax(1, round(c( sqrt(ncol(Xtr)), ncol(Xtr)/4, ncol(Xtr)/2 ))))
fit_rf <- train(
  x = Xtr, y = ytr,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_rf,
  ntree = 500
)
best_rf <- fit_rf$bestTune
auc_rf  <- max(fit_rf$results$ROC)

fit_svm <- train(
  x = Xtr, y = ytr,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)
best_svm <- fit_svm$bestTune
auc_svm  <- max(fit_svm$results$ROC)

grid_knn <- expand.grid(k = c(3,5,7,9,11,15,21))
fit_knn <- train(
  x = Xtr, y = ytr,
  method = "knn",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_knn
)
best_knn <- fit_knn$bestTune
auc_knn  <- max(fit_knn$results$ROC)

grid_gbm <- expand.grid(
  n.trees = c(200, 400, 800),
  interaction.depth = c(1, 2, 3),
  shrinkage = c(0.05, 0.1),
  n.minobsinnode = c(10, 20)
)
fit_gbm <- train(
  x = Xtr, y = ytr,
  method = "gbm",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_gbm,
  verbose = FALSE
)
best_gbm <- fit_gbm$bestTune
auc_gbm  <- max(fit_gbm$results$ROC)

grid_ada <- expand.grid(iter = c(100, 200, 400), maxdepth = c(1, 2, 3), nu = c(0.05, 0.1))
fit_ada <- train(
  x = Xtr, y = ytr,
  method = "ada",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_ada
)
best_ada <- fit_ada$bestTune
auc_ada  <- max(fit_ada$results$ROC)

set.seed(2025)
grid_xgb <- expand.grid(
  nrounds = c(300, 600),
  max_depth = c(3, 5, 7),
  eta = c(0.05, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 5),
  subsample = c(0.7, 1.0)
)
fit_xgb <- train(
  x = as.matrix(Xtr), y = ytr,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid_xgb,
  objective = "binary:logistic",
  eval_metric = "auc"
)
best_xgb <- fit_xgb$bestTune
auc_xgb  <- max(fit_xgb$results$ROC)

y_bin <- as.integer(ytr == "Yes")
dtrain_lgb <- lgb.Dataset(as.matrix(Xtr), label = y_bin)

lgb_param_grid <- expand.grid(
  num_leaves = c(15, 31, 63),
  max_depth = c(-1, 5, 7),
  learning_rate = c(0.05, 0.1),
  feature_fraction = c(0.8, 1.0),
  bagging_fraction = c(0.8, 1.0),
  bagging_freq = c(0, 1),
  lambda_l2 = c(0, 1)
)

cv_auc_lgb <- function(params) {
  p <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = params$num_leaves,
    max_depth = params$max_depth,
    learning_rate = params$learning_rate,
    feature_fraction = params$feature_fraction,
    bagging_fraction = params$bagging_fraction,
    bagging_freq = params$bagging_freq,
    lambda_l2 = params$lambda_l2,
    verbose = -1
  )
  cv <- lgb.cv(
    params = p,
    data = dtrain_lgb,
    nrounds = 2000,
    nfold = 5,
    early_stopping_rounds = 100,
    verbose = -1,
    stratified = TRUE
  )
  list(auc = max(unlist(cv$record_evals$valid$auc$eval)), best_iter = cv$best_iter)
}

lgb_results <- lapply(seq_len(nrow(lgb_param_grid)), function(i) {
  r <- cv_auc_lgb(lgb_param_grid[i, ])
  cbind(lgb_param_grid[i, ], data.frame(mean_auc = r$auc, best_iter = r$best_iter))
})
lgb_results <- do.call(rbind, lgb_results)
lgb_results <- lgb_results[order(-lgb_results$mean_auc), ]
best_lgb <- lgb_results[1, ]
auc_lgb  <- best_lgb$mean_auc

pool_tr <- catboost.load_pool(as.matrix(Xtr), label = y_bin)
cat_param_grid <- expand.grid(
  depth = c(4, 6, 8),
  learning_rate = c(0.05, 0.1),
  l2_leaf_reg = c(3, 5, 7),
  border_count = c(64, 128)
)

cv_auc_cat <- function(params) {
  p <- list(
    loss_function = "Logloss",
    eval_metric = "AUC",
    depth = params$depth,
    learning_rate = params$learning_rate,
    l2_leaf_reg = params$l2_leaf_reg,
    border_count = params$border_count,
    od_type = "Iter",
    od_wait = 100,
    random_seed = 2025,
    verbose = FALSE
  )
  cv <- catboost.cv(
    pool = pool_tr,
    params = p,
    fold_count = 5,
    type = "Classical",
    partition_random_seed = 2025,
    logging_level = "Silent",
    iterations = 2000
  )
  auc <- max(cv$test.AUC.mean)
  best_iter <- which.max(cv$test.AUC.mean)
  list(auc = auc, best_iter = best_iter)
}

cat_results <- lapply(seq_len(nrow(cat_param_grid)), function(i) {
  r <- cv_auc_cat(cat_param_grid[i, ])
  cbind(cat_param_grid[i, ], data.frame(mean_auc = r$auc, best_iter = r$best_iter))
})
cat_results <- do.call(rbind, cat_results)
cat_results <- cat_results[order(-cat_results$mean_auc), ]
best_cat <- cat_results[1, ]
auc_cat  <- best_cat$mean_auc

cv_summary <- rbind(
  data.frame(Model = "Logistic_ElasticNet", mean_auc = auc_glmnet, Params = paste0(capture.output(print(best_glmnet)), collapse=" ")),
  data.frame(Model = "RandomForest",        mean_auc = auc_rf,     Params = paste0(capture.output(print(best_rf)), collapse=" ")),
  data.frame(Model = "SVM_Radial",          mean_auc = auc_svm,    Params = paste0(capture.output(print(best_svm)), collapse=" ")),
  data.frame(Model = "KNN",                 mean_auc = auc_knn,    Params = paste0(capture.output(print(best_knn)), collapse=" ")),
  data.frame(Model = "GBM",                 mean_auc = auc_gbm,    Params = paste0(capture.output(print(best_gbm)), collapse=" ")),
  data.frame(Model = "AdaBoost",            mean_auc = auc_ada,    Params = paste0(capture.output(print(best_ada)), collapse=" ")),
  data.frame(Model = "XGBoost",             mean_auc = auc_xgb,    Params = paste0(capture.output(print(best_xgb)), collapse=" ")),
  data.frame(Model = "LightGBM",            mean_auc = auc_lgb,    Params = paste0(capture.output(print(best_lgb)), collapse=" ")),
  data.frame(Model = "CatBoost",            mean_auc = auc_cat,    Params = paste0(capture.output(print(best_cat)), collapse=" "))
)
cv_summary <- cv_summary[order(-cv_summary$mean_auc), ]
print(cv_summary)

final_glmnet <- glmnet::glmnet(x = as.matrix(Xtr), y = as.numeric(ytr=="Yes"),
                               alpha = best_glmnet$alpha, lambda = best_glmnet$lambda, family = "binomial", standardize = FALSE)
pred_glmnet_tr <- as.numeric(predict(final_glmnet, as.matrix(Xtr), type="response"))
pred_glmnet_te <- as.numeric(predict(final_glmnet, as.matrix(Xte), type="response"))

final_rf <- randomForest(x = Xtr, y = ytr, mtry = best_rf$mtry, ntree = 500)
pred_rf_tr <- as.numeric(predict(final_rf, Xtr, type="prob")[, "Yes"])
pred_rf_te <- as.numeric(predict(final_rf, Xte, type="prob")[, "Yes"])

final_svm <- e1071::svm(x = Xtr, y = ytr, kernel = "radial",
                        gamma = best_svm$sigma, cost = best_svm$C, probability = TRUE)
pred_svm_tr <- attr(predict(final_svm, Xtr, probability = TRUE), "probabilities")[, "Yes"]
pred_svm_te <- attr(predict(final_svm, Xte, probability = TRUE), "probabilities")[, "Yes"]

final_knn <- kknn::train.kknn(Result ~ ., data = data.frame(Result=ytr, Xtr), kmax = best_knn$k, kernel = "rectangular")
pred_knn_tr <- as.numeric(predict(final_knn, data.frame(Xtr), type = "prob")[, "Yes"])
pred_knn_te <- as.numeric(predict(final_knn, data.frame(Xte), type = "prob")[, "Yes"])

final_gbm <- gbm::gbm(
  formula = as.formula(paste("Result ~", paste(colnames(Xtr), collapse = "+"))),
  data    = data.frame(Result = ytr, Xtr),
  distribution = "bernoulli",
  n.trees = best_gbm$n.trees,
  interaction.depth = best_gbm$interaction.depth,
  shrinkage = best_gbm$shrinkage,
  n.minobsinnode = best_gbm$n.minobsinnode,
  verbose = FALSE
)
pred_gbm_tr <- as.numeric(predict(final_gbm, data.frame(Xtr), n.trees = best_gbm$n.trees, type = "response"))
pred_gbm_te <- as.numeric(predict(final_gbm, data.frame(Xte), n.trees = best_gbm$n.trees, type = "response"))

final_ada <- ada::ada(
  x = Xtr, y = ytr,
  iter = best_ada$iter,
  maxdepth = best_ada$maxdepth,
  nu = best_ada$nu
)
pred_ada_tr <- as.numeric(predict(final_ada, Xtr, type="prob")[, "Yes"])
pred_ada_te <- as.numeric(predict(final_ada, Xte, type="prob")[, "Yes"])

dtrain_xgb <- xgb.DMatrix(as.matrix(Xtr), label = as.numeric(ytr=="Yes"))
dtest_xgb  <- xgb.DMatrix(as.matrix(Xte))
params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_xgb$max_depth,
  eta = best_xgb$eta,
  gamma = best_xgb$gamma,
  colsample_bytree = best_xgb$colsample_bytree,
  min_child_weight = best_xgb$min_child_weight,
  subsample = best_xgb$subsample
)
final_xgb <- xgb.train(params_xgb, dtrain_xgb, nrounds = best_xgb$nrounds, verbose = 0)
pred_xgb_tr <- as.numeric(predict(final_xgb, dtrain_xgb))
pred_xgb_te <- as.numeric(predict(final_xgb, dtest_xgb))

dtrain_lgb_full <- lgb.Dataset(as.matrix(Xtr), label = as.numeric(ytr=="Yes"))
params_lgb <- list(
  objective = "binary",
  metric = "auc",
  num_leaves = best_lgb$num_leaves,
  max_depth = best_lgb$max_depth,
  learning_rate = best_lgb$learning_rate,
  feature_fraction = best_lgb$feature_fraction,
  bagging_fraction = best_lgb$bagging_fraction,
  bagging_freq = best_lgb$bagging_freq,
  lambda_l2 = best_lgb$lambda_l2,
  verbose = -1
)
final_lgb <- lgb.train(
  params = params_lgb,
  data = dtrain_lgb_full,
  nrounds = best_lgb$best_iter,
  verbose = -1
)
pred_lgb_tr <- as.numeric(predict(final_lgb, as.matrix(Xtr)))
pred_lgb_te <- as.numeric(predict(final_lgb, as.matrix(Xte)))

pool_tr_full <- catboost.load_pool(as.matrix(Xtr), label = as.numeric(ytr=="Yes"))
pool_te_full <- catboost.load_pool(as.matrix(Xte))
params_cat <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  depth = best_cat$depth,
  learning_rate = best_cat$learning_rate,
  l2_leaf_reg = best_cat$l2_leaf_reg,
  border_count = best_cat$border_count,
  iterations = best_cat$best_iter,
  verbose = FALSE,
  random_seed = 2025
)
final_cat <- catboost.train(pool = pool_tr_full, params = params_cat)
pred_cat_tr <- as.numeric(catboost.predict(final_cat, pool_tr_full, prediction_type = "Probability"))
pred_cat_te <- as.numeric(catboost.predict(final_cat, pool_te_full, prediction_type = "Probability"))

train_probe <- data.frame(
  ID = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
  GLMNET = pred_glmnet_tr,
  RF     = pred_rf_tr,
  SVM    = pred_svm_tr,
  KNN    = pred_knn_tr,
  GBM    = pred_gbm_tr,
  ADA    = pred_ada_tr,
  XGB    = pred_xgb_tr,
  LGB    = pred_lgb_tr,
  CAT    = pred_cat_tr
)
test_probe <- data.frame(
  ID = if ("ID" %in% names(test)) as.character(test$ID) else sprintf("test_%d", seq_len(nrow(test))),
  GLMNET = pred_glmnet_te,
  RF     = pred_rf_te,
  SVM    = pred_svm_te,
  KNN    = pred_knn_te,
  GBM    = pred_gbm_te,
  ADA    = pred_ada_te,
  XGB    = pred_xgb_te,
  LGB    = pred_lgb_te,
  CAT    = pred_cat_te
)
