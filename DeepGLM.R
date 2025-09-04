library(keras)
library(pROC)
library(missForest)
library(caret)
library(dplyr)

train <- dev
train$Result <- ifelse(train$Result == "Yes", 1, 0)

test <- vad[, var]
test$Result <- ifelse(test$Result == "Yes", 1, 0)

build_deepglm <- function(input_dim,
                          units = c(512, 256, 128),
                          dropouts = c(0.5, 0.4, 0.3),
                          l2_lambda = 0,
                          lr = 5e-4,
                          decay_steps = 1000,
                          decay_rate = 0.96) {
  input_layer <- layer_input(shape = input_dim)
  
  dnn_part <- input_layer %>%
    layer_dense(units = units[1], activation = "relu",
                kernel_initializer = "he_uniform",
                kernel_regularizer = if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropouts[1]) %>%
    layer_dense(units = units[2], activation = "relu",
                kernel_initializer = "he_uniform",
                kernel_regularizer = if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropouts[2]) %>%
    layer_dense(units = units[3], activation = "relu",
                kernel_regularizer = if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL) %>%
    layer_dropout(rate = dropouts[3])
  
  linear_part <- input_layer %>%
    layer_dense(units = 1, activation = "linear",
                kernel_regularizer = if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL)
  
  combined_output <- layer_add(list(dnn_part %>% layer_dense(units = 1, activation = "linear",
                                                             kernel_regularizer = if (l2_lambda > 0) regularizer_l2(l2_lambda) else NULL),
                                    linear_part)) %>%
    layer_activation("sigmoid")
  
  model <- keras_model(inputs = input_layer, outputs = combined_output)
  
  schedule <- learning_rate_schedule_exponential_decay(
    initial_learning_rate = lr,
    decay_steps = decay_steps,
    decay_rate = decay_rate,
    staircase = TRUE
  )
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = schedule),
    loss = "binary_crossentropy",
    metrics = list(metric_auc(name = "AUC"))
  )
  model
}

cv_mean_auc_for_params <- function(params, train_df, k = 5, epochs = 200, batch_size = 32, patience = 15) {
  folds <- createFolds(train_df$Result, k = k, list = TRUE, returnTrain = FALSE)
  aucs <- numeric(length(folds))
  for (i in seq_along(folds)) {
    val_idx <- folds[[i]]
    tr_idx  <- setdiff(seq_len(nrow(train_df)), val_idx)
    
    tr_df <- train_df[tr_idx, , drop = FALSE]
    vl_df <- train_df[val_idx, , drop = FALSE]
    
    tr_feat <- tr_df[, 2:ncol(tr_df)]
    vl_feat <- vl_df[, 2:ncol(vl_df)]
    tr_feat[] <- lapply(tr_feat, function(x) as.numeric(as.character(x)))
    vl_feat[] <- lapply(vl_feat, function(x) as.numeric(as.character(x)))
    
    imp_tr <- missForest(tr_feat)
    x_tr <- as.matrix(imp_tr$ximp)
    y_tr <- tr_df$Result
    
    imp_vl <- missForest(vl_feat)
    x_vl <- as.matrix(imp_vl$ximp)
    y_vl <- vl_df$Result
    
    k_clear_session()
    model <- build_deepglm(
      input_dim = ncol(x_tr),
      units = params$units[[1]],
      dropouts = params$dropouts[[1]],
      l2_lambda = params$l2,
      lr = params$lr,
      decay_steps = params$decay_steps,
      decay_rate = params$decay_rate
    )
    
    model %>% fit(
      x = x_tr, y = y_tr,
      validation_data = list(x_vl, y_vl),
      epochs = epochs, batch_size = batch_size,
      callbacks = list(callback_early_stopping(patience = patience, restore_best_weights = TRUE)),
      verbose = 0
    )
    
    pred_vl <- as.vector(predict(model, x_vl))
    aucs[i] <- as.numeric(pROC::auc(pROC::roc(y_vl, pred_vl, quiet = TRUE)))
  }
  mean(aucs, na.rm = TRUE)
}

units_grid <- list(c(512,256,128), c(256,128,64))
dropouts_grid <- list(c(0.5,0.4,0.3), c(0.4,0.3,0.2))
l2_grid <- c(0, 2e-4)
lr_grid <- c(5e-4, 1e-3)
batch_grid <- c(32, 64)
decay_rate_grid <- c(0.96, 0.9)

grid <- expand.grid(
  units = I(units_grid),
  dropouts = I(dropouts_grid),
  l2 = l2_grid,
  lr = lr_grid,
  batch_size = batch_grid,
  decay_steps = 1000,
  decay_rate = decay_rate_grid,
  stringsAsFactors = FALSE
)

grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))) {
  grid$mean_auc[i] <- cv_mean_auc_for_params(
    params = grid[i, ],
    train_df = train,
    k = 5,
    epochs = 200,
    batch_size = grid$batch_size[i],
    patience = 15
  )
}

grid <- grid %>% arrange(desc(mean_auc),
                         lengths(units),
                         l2, lr, batch_size)

best <- grid[1, , drop = FALSE]

train_features_full <- train[, 2:ncol(train)]
train_features_full[] <- lapply(train_features_full, function(x) as.numeric(as.character(x)))
imp_train_full <- missForest(train_features_full)
x_train <- as.matrix(imp_train_full$ximp)
y_train <- train$Result

test_features <- test[, 2:ncol(test)]
test_features[] <- lapply(test_features, function(x) as.numeric(as.character(x)))
imp_test <- missForest(test_features)
x_test <- as.matrix(imp_test$ximp)
y_test <- test$Result

k_clear_session()
final_model <- build_deepglm(
  input_dim = ncol(x_train),
  units = best$units[[1]],
  dropouts = best$dropouts[[1]],
  l2_lambda = best$l2,
  lr = best$lr,
  decay_steps = best$decay_steps,
  decay_rate = best$decay_rate
)

history <- final_model %>% fit(
  x = x_train, y = y_train,
  validation_split = 0.15,
  epochs = 200, batch_size = best$batch_size,
  callbacks = list(callback_early_stopping(patience = 15, restore_best_weights = TRUE)),
  verbose = 0
)

train_pred <- as.vector(predict(final_model, x_train))
test_pred  <- as.vector(predict(final_model, x_test))
