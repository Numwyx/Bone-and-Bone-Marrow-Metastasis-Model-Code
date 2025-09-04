library(reticulate)
library(deepregression)
library(keras)
library(tensorflow)
library(mgcv)
library(caret)
library(pROC)

set.seed(XXXX)
tf$random$set_seed(XXXX)

dev$Result <- ifelse(dev$Result == "Yes", 1L, 0L)
vad$Result <- ifelse(vad$Result == "Yes", 1L, 0L)

x_data <- as.data.frame(dev[, 2:ncol(dev)])
y_data <- dev$Result
x_test <- as.data.frame(vad[, 2:ncol(vad)])
y_test <- vad$Result

predictors <- names(x_data)
formula <- as.formula(paste("~", paste0("s(", predictors, ")", collapse = " + ")))

folds <- createFolds(y_data, k = 5, list = TRUE, returnTrain = TRUE)

param_grid <- expand.grid(
  lr         = c(1e-2, 1e-3, 5e-4, 1e-4),
  epochs     = c(30, 50, 100),
  batch_size = c(16, 32, 64, 128),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

results <- data.frame(lr=double(), epochs=integer(), batch_size=integer(), mean_auc=double())

for (i in seq_len(nrow(param_grid))) {
  lr_i     <- param_grid$lr[i]
  ep_i     <- param_grid$epochs[i]
  bs_i     <- param_grid$batch_size[i]
  
  auc_vec <- numeric(length(folds))
  
  for (f in seq_along(folds)) {
    tr_idx <- folds[[f]]
    va_idx <- setdiff(seq_len(nrow(x_data)), tr_idx)
    
    x_tr <- x_data[tr_idx, , drop = FALSE]
    y_tr <- y_data[tr_idx]
    x_va <- x_data[va_idx, , drop = FALSE]
    y_va <- y_data[va_idx]
    
    set.seed(2025); tf$random$set_seed(2025)
    
    model <- deepregression(
      y = y_tr,
      list_of_formulas = list(additive = formula),
      data = x_tr,
      family = "bernoulli",
      optimizer = optimizer_adam(learning_rate = lr_i)
    )
    
    fit(model, epochs = ep_i, batch_size = bs_i, verbose = 0)
    
    pred_va <- as.numeric(predict(model, newdata = x_va, type = "response"))
    if (length(unique(y_va)) < 2) {
      auc_vec[f] <- NA_real_
    } else {
      auc_vec[f] <- as.numeric(auc(roc(y_va, pred_va, quiet = TRUE)))
    }
  }
  
  results <- rbind(results, data.frame(
    lr = lr_i, epochs = ep_i, batch_size = bs_i,
    mean_auc = mean(auc_vec, na.rm = TRUE)
  ))
}

results <- results[order(-results$mean_auc, results$epochs, results$batch_size, results$lr), ]
print(results)

best_params <- results[1, ]
cat(sprintf("Best params -> lr=%.4g, epochs=%d, batch_size=%d | mean CV AUC=%.4f\n",
            best_params$lr, best_params$epochs, best_params$batch_size, best_params$mean_auc))

set.seed(2025); tf$random$set_seed(2025)
final_model <- deepregression(
  y = y_data,
  list_of_formulas = list(additive = formula),
  data = x_data,
  family = "bernoulli",
  optimizer = optimizer_adam(learning_rate = best_params$lr)
)
fit(final_model, epochs = best_params$epochs, batch_size = best_params$batch_size, verbose = 0)

train_probe <- data.frame(
  ID = if ("ID" %in% names(dev)) as.character(dev$ID) else sprintf("train_%d", seq_len(nrow(dev))),
  GAMNN = as.numeric(predict(final_model, newdata = x_data, type = "response"))
)
test_probe <- data.frame(
  ID = if ("ID" %in% names(vad)) as.character(vad$ID) else sprintf("test_%d", seq_len(nrow(vad))),
  GAMNN = as.numeric(predict(final_model, newdata = x_test, type = "response"))
)

train_auc <- as.numeric(auc(roc(y_data, train_probe$GAMNN, quiet = TRUE)))
test_auc  <- as.numeric(auc(roc(y_test,  test_probe$GAMNN,  quiet = TRUE)))

