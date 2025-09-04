library(keras)
library(tensorflow)
library(caret)
library(pROC)

set.seed(2025)
tf$random$set_seed(2025)

train <- dev
test  <- vad[, var]
train$Result <- ifelse(train$Result == "Yes", 1, 0)
test$Result  <- ifelse(test$Result  == "Yes", 1, 0)

x_train <- as.matrix(train[, 2:ncol(train), drop = FALSE])
y_train <- train$Result
x_test  <- as.matrix(test[,  2:ncol(test),  drop = FALSE])
y_test  <- test$Result

x_train <- scale(x_train)
ctr <- attr(x_train, "scaled:center"); scl <- attr(x_train, "scaled:scale")
x_test  <- scale(x_test, center = ctr, scale = scl)

pad_to_dim <- 28L * 28L
pad_features <- function(mat, target_dim = pad_to_dim) {
  p <- ncol(mat)
  if (p == target_dim) return(mat)
  if (p > target_dim) return(mat[, seq_len(target_dim), drop = FALSE])
  cbind(mat, matrix(0, nrow(mat), target_dim - p))
}
x_train_pad <- pad_features(x_train)
x_test_pad  <- pad_features(x_test)

x_train_img <- array_reshape(x_train_pad, c(nrow(x_train_pad), 28, 28, 1))
x_test_img  <- array_reshape(x_test_pad,  c(nrow(x_test_pad),  28, 28, 1))

build_cnn <- function(dropout = 0.3, l2_lambda = 3e-4, lr = 1e-3) {
  input <- layer_input(shape = c(28, 28, 1))
  x <- input %>%
    layer_conv_2d(32, 3, padding = "same",
                  activation = "relu",
                  kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(2) %>%
    layer_dropout(dropout) %>%
    layer_conv_2d(64, 3, padding = "same",
                  activation = "relu",
                  kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(2) %>%
    layer_dropout(dropout) %>%
    layer_flatten() %>%
    layer_dense(128, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_dropout(dropout)
  out <- layer_dense(x, 1, activation = "sigmoid")
  model <- keras_model(input, out)
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = c(metric_auc(name = "auc"))
  )
  model
}

dropout_grid <- c(0.2, 0.3)
l2_grid      <- c(1e-4, 3e-4)
lr_grid      <- c(1e-3, 5e-4)

folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = FALSE)
cv_results <- data.frame(dropout = numeric(), l2 = numeric(), lr = numeric(), mean_auc = numeric())

for (dp in dropout_grid) {
  for (l2v in l2_grid) {
    for (lr in lr_grid) {
      aucs <- numeric(length(folds))
      for (i in seq_along(folds)) {
        val_idx <- folds[[i]]
        tr_idx  <- setdiff(seq_len(nrow(x_train_img)), val_idx)
        
        x_tr <- x_train_img[tr_idx,,, , drop = FALSE]
        y_tr <- y_train[tr_idx]
        x_va <- x_train_img[val_idx,,, , drop = FALSE]
        y_va <- y_train[val_idx]
        
        model <- build_cnn(dropout = dp, l2_lambda = l2v, lr = lr)
        model %>% fit(
          x_tr, y_tr,
          epochs = 60, batch_size = 64,
          validation_data = list(x_va, y_va),
          verbose = 0,
          callbacks = list(
            callback_early_stopping(monitor = "val_auc", mode = "max",
                                    patience = 10, restore_best_weights = TRUE)
          )
        )
        ev <- model %>% evaluate(x_va, y_va, verbose = 0)
        aucs[i] <- as.numeric(ev[["auc"]])
        k_clear_session()
      }
      cv_results <- rbind(cv_results,
                          data.frame(dropout = dp, l2 = l2v, lr = lr,
                                     mean_auc = mean(aucs, na.rm = TRUE)))
    }
  }
}

cv_results <- cv_results[order(-cv_results$mean_auc, cv_results$dropout, cv_results$l2, cv_results$lr), ]
best <- cv_results[1, , drop = FALSE]

cat(sprintf("Best mean CV AUC = %.4f | dropout=%.2f, L2=%.4g, lr=%.4g\n",
            best$mean_auc, best$dropout, best$l2, best$lr))
print(cv_results)

final_model <- build_cnn(dropout = best$dropout, l2_lambda = best$l2, lr = best$lr)
final_model %>% fit(
  x_train_img, y_train,
  epochs = 60, batch_size = 64,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(
    callback_early_stopping(monitor = "val_auc", mode = "max",
                            patience = 10, restore_best_weights = TRUE)
  )
)

train_pred <- as.vector(final_model %>% predict(x_train_img))
test_pred  <- as.vector(final_model %>% predict(x_test_img))

auc_train <- as.numeric(pROC::auc(pROC::roc(y_train, train_pred, quiet = TRUE)))
auc_test  <- as.numeric(pROC::auc(pROC::roc(y_test,  test_pred,  quiet = TRUE)))
cat(sprintf("Final Train AUC: %.4f\n", auc_train))
cat(sprintf("Final Test  AUC: %.4f\n",  auc_test))

train_probe <- data.frame(ID = if ("ID" %in% names(train)) as.character(train$ID) else sprintf("train_%d", seq_len(nrow(train))),
                          CNN = train_pred)
test_probe  <- data.frame(ID = if ("ID" %in% names(test))  as.character(test$ID)  else sprintf("test_%d",  seq_len(nrow(test))),
                          CNN = test_pred)
