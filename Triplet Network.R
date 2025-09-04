library(keras)
library(tensorflow)
library(dplyr)
library(caret)
library(pROC)
library(FNN)

set.seed(XXXX)
tf$random$set_seed(XXXX)

dev$Result <- ifelse(dev$Result == "Yes", 1L, 0L)
vad$Result <- ifelse(vad$Result == "Yes", 1L, 0L)

x_all <- as.matrix(dev[, setdiff(names(dev), "Result"), drop = FALSE])
y_all <- as.integer(dev$Result)
x_vad <- as.matrix(vad[, setdiff(names(vad), "Result"), drop = FALSE])
y_vad <- as.integer(vad$Result)

embedding_dim <- 32

create_base_network <- function(input_shape, units1 = 64, embed_dim = 32, dropout = 0.3, l2_lambda = 3e-4) {
  input <- layer_input(shape = input_shape)
  x <- input %>%
    layer_dense(units = units1, activation = "relu",
                kernel_regularizer = regularizer_l2(l2_lambda)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = embed_dim,
                kernel_regularizer = regularizer_l2(l2_lambda))
  keras_model(input, x)
}

triplet_loss_builder <- function(alpha = 0.2, embed_dim = 32) {
  function(y_true, y_pred) {
    a <- y_pred[, 1:embed_dim]
    p <- y_pred[, (embed_dim + 1):(2 * embed_dim)]
    n <- y_pred[, (2 * embed_dim + 1):(3 * embed_dim)]
    pos <- k_sum(k_square(a - p), axis = -1)
    neg <- k_sum(k_square(a - n), axis = -1)
    k_mean(k_maximum(pos - neg + alpha, 0))
  }
}

generate_triplets <- function(features, labels, num_triplets = 3000) {
  anchors <- positives <- negatives <- vector("list", num_triplets)
  for (i in seq_len(num_triplets)) {
    ai <- sample.int(nrow(features), 1)
    yl <- labels[ai]
    pi <- sample(which(labels == yl), 1)
    ni <- sample(which(labels != yl), 1)
    anchors[[i]]  <- features[ai, , drop = FALSE]
    positives[[i]]<- features[pi, , drop = FALSE]
    negatives[[i]]<- features[ni, , drop = FALSE]
  }
  list(
    anchor   = do.call(rbind, anchors),
    positive = do.call(rbind, positives),
    negative = do.call(rbind, negatives)
  )
}

knn_predict_prob <- function(train_embeds, train_labels, test_embeds, k = 5) {
  nn <- FNN::get.knnx(train_embeds, test_embeds, k = k)$nn.index
  sapply(seq_len(nrow(nn)), function(i) mean(train_labels[nn[i, ]]))
}

cv_mean_auc_for <- function(x_all, y_all,
                            units = 64, margin = 0.2, k_val = 5,
                            dropout = 0.3, l2_lambda = 3e-4, lr = 1e-3,
                            folds = 5, epochs = 20, triplets_n = 3000) {
  idx_list <- createFolds(y_all, k = folds, list = TRUE, returnTrain = FALSE)
  aucs <- numeric(length(idx_list))
  for (i in seq_along(idx_list)) {
    val_idx <- idx_list[[i]]
    tr_idx  <- setdiff(seq_len(nrow(x_all)), val_idx)
    
    pre <- preProcess(x_all[tr_idx, , drop = FALSE], method = c("center", "scale"))
    x_tr <- as.matrix(predict(pre, x_all[tr_idx, , drop = FALSE]))
    x_va <- as.matrix(predict(pre, x_all[val_idx, , drop = FALSE]))
    y_tr <- y_all[tr_idx]
    y_va <- y_all[val_idx]
    
    base_network <- create_base_network(ncol(x_all), units1 = units,
                                        embed_dim = embedding_dim,
                                        dropout = dropout, l2_lambda = l2_lambda)
    
    ai <- layer_input(shape = ncol(x_all))
    pi <- layer_input(shape = ncol(x_all))
    ni <- layer_input(shape = ncol(x_all))
    
    ea <- base_network(ai); ep <- base_network(pi); en <- base_network(ni)
    merged <- layer_concatenate(list(ea, ep, en), axis = 1)
    triplet_model <- keras_model(inputs = list(ai, pi, ni), outputs = merged)
    triplet_model %>% compile(optimizer = optimizer_adam(learning_rate = lr),
                              loss = triplet_loss_builder(alpha = margin, embed_dim = embedding_dim))
    
    trip <- generate_triplets(x_tr, y_tr, num_triplets = triplets_n)
    triplet_model %>% fit(
      x = list(trip$anchor, trip$positive, trip$negative),
      y = matrix(0, nrow = triplets_n, ncol = 3 * embedding_dim),
      batch_size = 32, epochs = epochs, verbose = 0
    )
    
    emb_model <- keras_model(inputs = base_network$input, outputs = base_network$output)
    emb_tr <- predict(emb_model, x_tr)
    emb_va <- predict(emb_model, x_va)
    pr_va  <- knn_predict_prob(emb_tr, y_tr, emb_va, k = k_val)
    aucs[i] <- as.numeric(auc(roc(y_va, pr_va, quiet = TRUE)))
    k_clear_session()
  }
  mean(aucs, na.rm = TRUE)
}

dropout_grid <- c(0.2, 0.3)
l2_grid      <- c(1e-4, 3e-4)
lr_grid      <- c(1e-3, 5e-4)

grid <- expand.grid(
  dropout = dropout_grid,
  l2      = l2_grid,
  lr      = lr_grid,
  stringsAsFactors = FALSE
)
grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))) {
  grid$mean_auc[i] <- cv_mean_auc_for(
    x_all, y_all,
    units = 64, margin = 0.2, k_val = 5,
    dropout = grid$dropout[i],
    l2_lambda = grid$l2[i],
    lr = grid$lr[i],
    folds = 5, epochs = 20, triplets_n = 3000
  )
}

grid <- grid[order(-grid$mean_auc, grid$dropout, grid$l2, grid$lr), ]
best <- grid[1, , drop = FALSE]
print(grid)
cat(sprintf("Best params -> dropout=%.2f, L2=%.4g, lr=%.4g | mean CV AUC=%.4f\n",
            best$dropout, best$l2, best$lr, best$mean_auc))

pre_full <- preProcess(x_all, method = c("center", "scale"))
x_all_s  <- as.matrix(predict(pre_full, x_all))
x_vad_s  <- as.matrix(predict(pre_full, x_vad))

base_network <- create_base_network(ncol(x_all_s), units1 = 64,
                                    embed_dim = embedding_dim,
                                    dropout = best$dropout,
                                    l2_lambda = best$l2)

ai <- layer_input(shape = ncol(x_all_s))
pi <- layer_input(shape = ncol(x_all_s))
ni <- layer_input(shape = ncol(x_all_s))

ea <- base_network(ai); ep <- base_network(pi); en <- base_network(ni)
merged <- layer_concatenate(list(ea, ep, en), axis = 1)
triplet_model <- keras_model(inputs = list(ai, pi, ni), outputs = merged)
triplet_model %>% compile(optimizer = optimizer_adam(learning_rate = best$lr),
                          loss = triplet_loss_builder(alpha = 0.2, embed_dim = embedding_dim))

final_trip <- generate_triplets(x_all_s, y_all, num_triplets = 5000)
triplet_model %>% fit(
  x = list(final_trip$anchor, final_trip$positive, final_trip$negative),
  y = matrix(0, nrow = 5000, ncol = 3 * embedding_dim),
  batch_size = 32, epochs = 25, verbose = 0
)

emb_model <- keras_model(inputs = base_network$input, outputs = base_network$output)
emb_tr <- predict(emb_model, x_all_s)
emb_va <- predict(emb_model, x_vad_s)

train_prob <- knn_predict_prob(emb_tr, y_all, emb_tr, k = 5)
test_prob  <- knn_predict_prob(emb_tr, y_all, emb_va, k = 5)

auc_tr <- as.numeric(auc(roc(y_all, train_prob, quiet = TRUE)))
auc_te <- as.numeric(auc(roc(y_vad, test_prob,  quiet = TRUE)))
cat(sprintf("Final Train AUC: %.4f\n", auc_tr))
cat(sprintf("Final Test  AUC: %.4f\n",  auc_te))

train_probe <- data.frame(
  ID = if ("ID" %in% names(dev)) as.character(dev$ID) else sprintf("train_%d", seq_len(nrow(dev))),
  MetricLearning = train_prob
)
test_probe <- data.frame(
  ID = if ("ID" %in% names(vad)) as.character(vad$ID) else sprintf("test_%d", seq_len(nrow(vad))),
  MetricLearning = test_prob
)
