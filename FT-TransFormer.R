library(keras)
library(tensorflow)
library(caret)

train <- dev
train$Result <- ifelse(train$Result == "Yes", 1, 0)
X <- as.matrix(train[, 2:ncol(train), drop = FALSE])
y <- train$Result

ft_block <- function(x, d_model, n_heads, dropout_rate, l2_lambda) {
  attn <- layer_multi_head_attention(num_heads = n_heads, key_dim = as.integer(d_model / n_heads))
  a <- attn(query = x, value = x, key = x)
  a <- layer_dropout(a, rate = dropout_rate)
  x <- layer_add(list(x, a))
  x <- layer_layer_normalization(x)
  
  f <- layer_dense(x, units = d_model * 4, activation = "gelu",
                   kernel_regularizer = regularizer_l2(l2_lambda))
  f <- layer_dropout(f, rate = dropout_rate)
  f <- layer_dense(f, units = d_model,
                   kernel_regularizer = regularizer_l2(l2_lambda))
  f <- layer_dropout(f, rate = dropout_rate)
  x <- layer_add(list(x, f))
  layer_layer_normalization(x)
}

build_ftt <- function(input_dim,
                      n_tokens = 8L,
                      d_model  = 64L,
                      n_blocks = 2L,
                      n_heads  = 4L,
                      dropout_rate = 0.3,
                      l2_lambda   = 3e-4,
                      lr = 1e-3) {
  stopifnot(d_model %% n_heads == 0)
  
  inp <- layer_input(shape = input_dim)
  
  tok <- inp %>%
    layer_dense(units = n_tokens * d_model, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_reshape(target_shape = c(n_tokens, d_model))
  
  x <- tok
  for (i in seq_len(n_blocks)) {
    x <- ft_block(x, d_model = d_model, n_heads = n_heads,
                  dropout_rate = dropout_rate, l2_lambda = l2_lambda)
  }
  
  x <- layer_global_average_pooling_1d(x) %>%
    layer_dense(units = d_model, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_dropout(rate = dropout_rate)
  
  out <- layer_dense(x, units = 1, activation = "sigmoid")
  
  model <- keras_model(inp, out)
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = c(metric_auc(name = "auc"))
  )
  model
}


grid <- expand.grid(
  d_model  = c(64L, 128L),
  dropout  = c(0.2, 0.3),
  l2       = c(3e-4, 1e-4),
  lr       = c(1e-3, 5e-4),
  n_tokens = c(8L),
  n_blocks = c(2L),
  n_heads  = c(4L),
  stringsAsFactors = FALSE
)


folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
results <- data.frame()

for (i in seq_len(nrow(grid))) {
  gi <- grid[i, ]
  aucs <- numeric(length(folds))
  
  for (f in seq_along(folds)) {
    val_idx <- folds[[f]]
    tr_idx  <- setdiff(seq_len(nrow(X)), val_idx)
    
    x_tr <- X[tr_idx, , drop = FALSE]; y_tr <- y[tr_idx]
    x_va <- X[val_idx, , drop = FALSE]; y_va <- y[val_idx]
    
    model <- build_ftt(
      input_dim   = ncol(X),
      n_tokens    = gi$n_tokens,
      d_model     = gi$d_model,
      n_blocks    = gi$n_blocks,
      n_heads     = gi$n_heads,
      dropout_rate= gi$dropout,
      l2_lambda   = gi$l2,
      lr          = gi$lr
    )
    
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
    aucs[f] <- as.numeric(ev[["auc"]])
    k_clear_session()
  }
  
  results <- rbind(results, cbind(gi, mean_auc = mean(aucs, na.rm = TRUE)))
}

results <- results[order(-results$mean_auc, results$d_model, results$dropout, results$l2, results$lr), ]
best <- results[1, , drop = FALSE]

cat(sprintf("Best mean CV AUC = %.4f | d_model=%d, dropout=%.2f, l2=%.4g, lr=%.4g, tokens=%d, blocks=%d, heads=%d\n",
            best$mean_auc, best$d_model, best$dropout, best$l2, best$lr,
            best$n_tokens, best$n_blocks, best$n_heads))


final_model <- build_ftt(
  input_dim   = ncol(X),
  n_tokens    = best$n_tokens,
  d_model     = best$d_model,
  n_blocks    = best$n_blocks,
  n_heads     = best$n_heads,
  dropout_rate= best$dropout,
  l2_lambda   = best$l2,
  lr          = best$lr
)

hist_final <- final_model %>% fit(
  X, y,
  epochs = 60, batch_size = 64,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(
    callback_early_stopping(monitor = "val_auc", mode = "max",
                            patience = 10, restore_best_weights = TRUE)
  )
)


test <- vad[, var]
test$Result <- ifelse(test$Result == "Yes", 1, 0)
X_te <- as.matrix(test[, 2:ncol(test), drop = FALSE])

pred_train <- as.vector(final_model %>% predict(X))
pred_test  <- as.vector(final_model %>% predict(X_te))

auc_train <- as.numeric(pROC::auc(pROC::roc(y, pred_train, quiet = TRUE)))
auc_test  <- as.numeric(pROC::auc(pROC::roc(test$Result, pred_test, quiet = TRUE)))

train_probe <- data.frame(ID = rownames(train), FT_Transformer = pred_train)
test_probe  <- data.frame(ID = rownames(test),  FT_Transformer = pred_test)
