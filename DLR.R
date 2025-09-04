library(keras)
library(tensorflow)
library(caret)
library(pROC)

set.seed(2025)
tf$random$set_seed(2025)

train <- dev
train$Result <- ifelse(train$Result == "Yes", 1L, 0L)
x_train <- as.matrix(train[, setdiff(names(train), "Result"), drop = FALSE])
y_train <- train$Result

test <- vad[, var]
test$Result <- ifelse(test$Result == "Yes", 1L, 0L)
x_test <- as.matrix(test[, setdiff(names(test), "Result"), drop = FALSE])
y_test <- test$Result

build_dlr <- function(input_dim, dropout1 = 0.3, dropout2 = 0.2, l2_lambda = 3e-4, lr = 1e-3) {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = input_dim,
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_dropout(rate = dropout1) %>%
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_dropout(rate = dropout2) %>%
    layer_dense(units = 1, activation = "sigmoid")
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "binary_crossentropy",
    metrics = list(metric_auc(name = "auc"))
  )
  model
}

cv_mean_auc <- function(x, y, params, k = 5, epochs = 50, batch_size = 32, patience = 10) {
  folds <- createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
  aucs <- numeric(length(folds))
  for (i in seq_along(folds)) {
    val_idx <- folds[[i]]
    tr_idx  <- setdiff(seq_len(nrow(x)), val_idx)
    
    x_tr <- x[tr_idx, , drop = FALSE]
    y_tr <- y[tr_idx]
    x_va <- x[val_idx, , drop = FALSE]
    y_va <- y[val_idx]
    
    k_clear_session()
    model <- build_dlr(
      input_dim  = ncol(x),
      dropout1   = params$dropout1,
      dropout2   = params$dropout2,
      l2_lambda  = params$l2,
      lr         = params$lr
    )
    
    model %>% fit(
      x = x_tr, y = y_tr,
      validation_data = list(x_va, y_va),
      epochs = epochs, batch_size = batch_size,
      callbacks = list(callback_early_stopping(monitor = "val_auc", mode = "max",
                                               patience = patience, restore_best_weights = TRUE)),
      verbose = 0
    )
    
    p_va <- as.numeric(predict(model, x_va))
    aucs[i] <- as.numeric(pROC::auc(pROC::roc(y_va, p_va, quiet = TRUE)))
  }
  mean(aucs, na.rm = TRUE)
}

grid <- expand.grid(
  dropout1 = c(0.3, 0.2),
  dropout2 = c(0.2, 0.1),
  l2       = c(1e-4, 3e-4),
  lr       = c(1e-3, 5e-4),
  stringsAsFactors = FALSE
)
grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))) {
  grid$mean_auc[i] <- cv_mean_auc(
    x_train, y_train,
    params = grid[i, ],
    k = 5, epochs = 50, batch_size = 32, patience = 10
  )
}

grid <- grid[order(-grid$mean_auc, grid$dropout1, grid$dropout2, grid$l2, grid$lr), ]
print(grid)
best <- grid[1, , drop = FALSE]
cat(sprintf("Best params -> dropout1=%.2f, dropout2=%.2f, L2=%.4g, lr=%.4g | mean CV AUC=%.4f\n",
            best$dropout1, best$dropout2, best$l2, best$lr, best$mean_auc))

k_clear_session()
final_model <- build_dlr(
  input_dim  = ncol(x_train),
  dropout1   = best$dropout1,
  dropout2   = best$dropout2,
  l2_lambda  = best$l2,
  lr         = best$lr
)

history <- final_model %>% fit(
  x = x_train, y = y_train,
  validation_split = 0.15,
  epochs = 50, batch_size = 32,
  callbacks = list(callback_early_stopping(monitor = "val_auc", mode = "max",
                                           patience = 10, restore_best_weights = TRUE)),
  verbose = 0
)

train_pred <- as.numeric(predict(final_model, x_train))
test_pred  <- as.numeric(predict(final_model, x_test))

train_probe <- data.frame(
  ID  = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
  DLR = train_pred
)
test_probe <- data.frame(
  ID  = if ("ID" %in% names(test))  as.character(test$ID)  else sprintf("test_%d",  seq_len(nrow(test))),
  DLR = test_pred
)
