library(keras)
library(caret)
library(pROC)
library(dplyr)
library(readr)

dev_path <- "dev.csv"
vad_path <- "vad.csv"
dev <- readr::read_csv(dev_path, show_col_types = FALSE)
vad <- readr::read_csv(vad_path, show_col_types = FALSE)

var <- c("Weight","Charlson_Comorbidity_Index","SOFA","Heart_Rate","Resp_Rate","Lactate","Hematocrit","Calcium","Potassium","WBC","Albumin")

if (!"Result" %in% names(dev)) stop("dev 缺少 Result 列")
if (!"Result" %in% names(vad)) stop("vad 缺少 Result 列")

common_vars <- intersect(var, intersect(names(dev), names(vad)))
if (length(common_vars) == 0) stop("所给 var 在 dev 与 vad 中无交集")
dev <- dev[, c("Result", common_vars)]
vad <- vad[, c("Result", common_vars)]

id_train <- if ("ID" %in% names(dev)) as.character(dev$ID) else sprintf("train_%d", seq_len(nrow(dev)))
id_test  <- if ("ID" %in% names(vad)) as.character(vad$ID) else sprintf("test_%d", seq_len(nrow(vad)))

train <- dev
test  <- vad

train$Result <- ifelse(train$Result == "Yes", 1, ifelse(train$Result == "No", 0, train$Result))
test$Result  <- ifelse(test$Result  == "Yes", 1, ifelse(test$Result  == "No", 0, test$Result))

prep_recipe <- function(df_fit, df_apply) {
  y_fit <- df_fit$Result
  X_fit <- df_fit %>% dplyr::select(-Result)
  X_fit <- X_fit %>% mutate(across(where(is.character), ~replace(., is.na(.), "Missing")))
  X_fit <- X_fit %>% mutate(across(where(is.logical), ~as.character(.))) %>% mutate(across(where(is.character), ~factor(.)))
  dv <- dummyVars(~ ., data = X_fit, fullRank = TRUE)
  X_fit_mm <- predict(dv, newdata = X_fit) %>% as.data.frame()
  pp <- preProcess(X_fit_mm, method = c("medianImpute","center","scale"))
  X_fit_scaled <- predict(pp, X_fit_mm) %>% as.matrix()
  y_apply <- df_apply$Result
  X_apply <- df_apply %>% dplyr::select(-Result)
  X_apply <- X_apply %>% mutate(across(where(is.character), ~replace(., is.na(.), "Missing")))
  X_apply <- X_apply %>% mutate(across(where(is.logical), ~as.character(.))) %>% mutate(across(where(is.character), ~factor(.)))
  X_apply_mm <- predict(dv, newdata = X_apply) %>% as.data.frame()
  X_apply_scaled <- predict(pp, X_apply_mm) %>% as.matrix()
  list(X_fit = X_fit_scaled, y_fit = y_fit, X_apply = X_apply_scaled, y_apply = y_apply, dv = dv, pp = pp)
}

build_mlp <- function(input_dim, hidden_units = c(128,64,32), dropout_rate = 0.3, l2_lambda = 1e-4, lr = 1e-3) {
  model <- keras_model_sequential()
  for (i in seq_along(hidden_units)) {
    if (i == 1) {
      model %>% layer_dense(units = hidden_units[i], input_shape = input_dim, activation = "relu", kernel_regularizer = regularizer_l2(l2_lambda)) %>% layer_batch_normalization() %>% layer_dropout(rate = dropout_rate)
    } else {
      model %>% layer_dense(units = hidden_units[i], activation = "relu", kernel_regularizer = regularizer_l2(l2_lambda)) %>% layer_batch_normalization() %>% layer_dropout(rate = dropout_rate)
    }
  }
  model %>% layer_dense(units = 1, activation = "sigmoid")
  model %>% compile(optimizer = optimizer_adam(learning_rate = lr), loss = "binary_crossentropy", metrics = list(metric_auc(name = "AUC")))
  model
}

grid <- expand.grid(
  arch = I(list(c(64,32), c(128,64,32), c(64,64,32))),
  dropout = c(0.2, 0.3),
  l2 = c(1e-4, 3e-4),
  lr = c(1e-3, 3e-4),
  batch_size = c(32, 64),
  epochs = c(100)
)

folds <- caret::createFolds(train$Result, k = 5, list = TRUE, returnTrain = FALSE)

cv_score <- function(params, train_df) {
  aucs <- c()
  for (fold_idx in seq_along(folds)) {
    val_idx <- folds[[fold_idx]]
    tr_idx  <- setdiff(seq_len(nrow(train_df)), val_idx)
    tr_df <- train_df[tr_idx, , drop = FALSE]
    va_df <- train_df[val_idx, , drop = FALSE]
    prep_cv <- prep_recipe(tr_df, va_df)
    model <- build_mlp(input_dim = ncol(prep_cv$X_fit), hidden_units = params$arch[[1]], dropout_rate = params$dropout, l2_lambda = params$l2, lr = params$lr)
    cb_es <- callback_early_stopping(monitor = "val_auc", mode = "max", patience = 15, restore_best_weights = TRUE)
    cb_rlr <- callback_reduce_lr_on_plateau(monitor = "val_auc", mode = "max", factor = 0.5, patience = 7, min_lr = 1e-6)
    model %>% fit(x = prep_cv$X_fit, y = prep_cv$y_fit, validation_data = list(prep_cv$X_apply, prep_cv$y_apply), epochs = params$epochs, batch_size = params$batch_size, verbose = 0, callbacks = list(cb_es, cb_rlr))
    pred_val <- as.numeric(model %>% predict(prep_cv$X_apply))
    roc_obj <- pROC::roc(prep_cv$y_apply, pred_val, quiet = TRUE)
    aucs <- c(aucs, as.numeric(pROC::auc(roc_obj)))
    k_clear_session()
  }
  mean(aucs, na.rm = TRUE)
}

grid$mean_auc <- NA_real_
for (i in seq_len(nrow(grid))) {
  grid$mean_auc[i] <- cv_score(grid[i, ], train)
}

grid <- grid %>% arrange(desc(mean_auc), lengths(arch), dropout, l2, batch_size)
best <- grid[1, ]

prep_final <- prep_recipe(train, test)
model_final <- build_mlp(input_dim = ncol(prep_final$X_fit), hidden_units = best$arch[[1]], dropout_rate = best$dropout, l2_lambda = best$l2, lr = best$lr)

cb_es <- callback_early_stopping(monitor = "val_auc", mode = "max", patience = 20, restore_best_weights = TRUE)
cb_rlr <- callback_reduce_lr_on_plateau(monitor = "val_auc", mode = "max", factor = 0.5, patience = 8, min_lr = 1e-6)

history_final <- model_final %>% fit(x = prep_final$X_fit, y = prep_final$y_fit, validation_split = 0.15, epochs = best$epochs, batch_size = best$batch_size, verbose = 0, callbacks = list(cb_es, cb_rlr))

train_pred <- as.numeric(model_final %>% predict(prep_final$X_fit))
test_pred  <- as.numeric(model_final %>% predict(prep_final$X_apply))

train_probe <- data.frame(ID = id_train, MLP = train_pred)
test_probe  <- data.frame(ID = id_test,  MLP = test_pred)
