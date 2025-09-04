library(h2o)
library(pROC)
library(precrec)

h2o.init(nthreads = -1)

train <- dev
test  <- vad[, var]
train$Result <- as.factor(train$Result)
test$Result  <- as.factor(test$Result)

train_h2o <- as.h2o(train)
test_h2o  <- as.h2o(test)

response   <- "Result"
predictors <- setdiff(colnames(train_h2o), response)

hyper_params <- list(
  hidden = list(c(64,32), c(128,64,32)),
  rate = c(0.001, 0.0005),
  l2 = c(1e-4, 3e-4),
  input_dropout_ratio = c(0.0, 0.1),
  hidden_dropout_ratios = list(c(0.5,0.5), c(0.5,0.4,0.3))
)

grid_id <- "dl_cv_grid"
h2o.grid(
  algorithm = "deeplearning",
  grid_id = grid_id,
  x = predictors,
  y = response,
  training_frame = train_h2o,
  activation = "RectifierWithDropout",
  epochs = 100,
  nfolds = 5,
  fold_assignment = "Stratified",
  stopping_metric = "AUC",
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  balance_classes = TRUE,
  variable_importances = TRUE,
  seed = 123,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian")
)

g <- h2o.getGrid(grid_id, sort_by = "auc", decreasing = TRUE)
best_id <- g@model_ids[[1]]
best_cv_model <- h2o.getModel(best_id)
bp <- best_cv_model@allparameters

final_model <- h2o.deeplearning(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  activation = bp$activation,
  hidden = bp$hidden,
  epochs = 200,
  rate = bp$rate,
  l1 = ifelse(is.null(bp$l1), 0, bp$l1),
  l2 = bp$l2,
  input_dropout_ratio = ifelse(is.null(bp$input_dropout_ratio), 0, bp$input_dropout_ratio),
  hidden_dropout_ratios = bp$hidden_dropout_ratios,
  stopping_metric = "AUC",
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  balance_classes = TRUE,
  seed = 123
)

train_pred_h2o <- h2o.predict(final_model, train_h2o)
test_pred_h2o  <- h2o.predict(final_model, test_h2o)