library(keras)
library(tensorflow)
library(caret)
library(pROC)

set.seed(XXXX); tf$random$set_seed(XXXX)

train <- dev; test <- vad[, var]
train$Result <- ifelse(train$Result=="Yes",1L,0L)
test$Result  <- ifelse(test$Result=="Yes",1L,0L)

Xtr <- as.matrix(train[, -1, drop = FALSE])
ytr <- train$Result
Xte <- as.matrix(test[,  -1, drop = FALSE])
yte <- test$Result

Xtr <- scale(Xtr); ctr <- attr(Xtr,"scaled:center"); scl <- attr(Xtr,"scaled:scale")
Xte <- scale(Xte, center = ctr, scale = scl)

pad_to <- 28L*28L
padf <- function(m,t=pad_to){p<-ncol(m); if(p==t) m else if(p>t) m[,1:t,drop=FALSE] else cbind(m, matrix(0,nrow(m),t-p))}
Xtr <- array_reshape(padf(Xtr), c(nrow(Xtr),28,28,1))
Xte <- array_reshape(padf(Xte),  c(nrow(Xte),28,28,1))

latent_dim <- 64L

label_map <- function(y, h=28L, w=28L){
  y <- array(as.numeric(y), dim = c(length(y), 1, 1, 1))
  k_repeat(k_repeat(y, h, axis = 2L), w, axis = 3L)
}

build_generator <- function(latent_dim=64L){
  z  <- layer_input(shape = latent_dim)
  yc <- layer_input(shape = 1L)
  emb <- yc %>% layer_embedding(input_dim = 2L, output_dim = 8L) %>% layer_flatten()
  
  x <- list(z, emb) %>% layer_concatenate() %>%
    layer_dense(128*7*7) %>% layer_activation("relu") %>%
    layer_reshape(c(7,7,128)) %>%
    layer_conv_2d_transpose(64,5,strides=2,padding="same",activation="relu") %>%
    layer_conv_2d_transpose(1,5,strides=2,padding="same",activation="tanh")
  keras_model(list(z,yc), x)
}

build_discriminator <- function(dropout=0.3, l2=3e-4, lr=2e-4){
  xin <- layer_input(shape = c(28,28,1))
  yc  <- layer_input(shape = 1L)
  ymap <- layer_lambda(yc, function(y) label_map(y))  # (n,28,28,1)
  xcat <- layer_concatenate(list(xin, ymap), axis = -1L)
  
  x <- xcat %>%
    layer_conv_2d(64,5,strides=2,padding="same",activation="relu",
                  kernel_regularizer=regularizer_l2(l2)) %>%
    layer_dropout(dropout) %>%
    layer_conv_2d(128,5,strides=2,padding="same",activation="relu",
                  kernel_regularizer=regularizer_l2(l2)) %>%
    layer_dropout(dropout) %>%
    layer_flatten() %>%
    layer_dense(128, activation="relu", name="feat",
                kernel_regularizer=regularizer_l2(l2))
  out <- layer_dense(x, 3, activation="softmax")  # 0=real-0, 1=real-1, 2=fake
  
  d <- keras_model(list(xin,yc), out)
  d %>% compile(optimizer = optimizer_adam(learning_rate=lr, beta_1=0.5),
                loss = "sparse_categorical_crossentropy")
  # 暴露特征子模型：输入(样本, 标签) -> feat
  feat_model <- keras_model(d$input, get_layer(d, "feat")$output)
  list(d=d, feat=feat_model)
}

ssgan_step <- function(g, d, feat_mdl, batch_real_x, batch_real_y,
                       batch_size = 64L, latent_dim = 64L){
  # 1) real
  lab_real <- ifelse(batch_real_y==0L, 0L, 1L)
  d$d %>% train_on_batch(list(batch_real_x, array(batch_real_y, dim=c(length(batch_real_y),1))), lab_real)
  
  # 2) fake
  z <- matrix(rnorm(batch_size*latent_dim), ncol=latent_dim)
  y_fake <- sample(0:1, size = nrow(z), replace = TRUE)
  x_fake <- predict(g, list(z, matrix(y_fake, ncol=1)))
  lab_fake <- array(2L, dim=c(nrow(z),1))
  d$d %>% train_on_batch(list(x_fake, matrix(y_fake, ncol=1)), lab_fake)
  
  # a) adversarial
  freeze_weights(d$d)
  z2 <- matrix(rnorm(batch_size*latent_dim), ncol=latent_dim)
  y2 <- sample(0:1, size = nrow(z2), replace = TRUE)
  gan <- keras_model(list(g$input[[1]], g$input[[2]]), d$d(list(g$output, g$input[[2]])))
  gan %>% compile(optimizer=optimizer_adam(learning_rate=2e-4, beta_1=0.5),
                  loss="sparse_categorical_crossentropy")
  lab_trick <- array(sample(c(0L,1L), nrow(z2), replace=TRUE), dim=c(nrow(z2),1))
  gan %>% train_on_batch(list(z2, matrix(y2,ncol=1)), lab_trick)
  unfreeze_weights(d$d)
  
  # b) Feature Matching
  real_feat <- predict(feat_mdl, list(batch_real_x, array(batch_real_y, dim=c(length(batch_real_y),1))))
  mu_real <- matrix(colMeans(real_feat), nrow = nrow(z), ncol = ncol(real_feat), byrow = TRUE)
  fm <- keras_model(list(g$input[[1]], g$input[[2]]), feat_mdl(list(g$output, g$input[[2]])))
  fm %>% compile(optimizer = optimizer_adam(learning_rate=2e-4, beta_1=0.5), loss = "mse")
  z3 <- matrix(rnorm(batch_size*latent_dim), ncol=latent_dim)
  y3 <- sample(0:1, size = nrow(z3), replace = TRUE)
  fm %>% train_on_batch(list(z3, matrix(y3,ncol=1)), mu_real[seq_len(nrow(z3)), , drop=FALSE])
}

train_cssgan <- function(X, y, epochs=60, batch_size=64, dropout=0.3, l2=3e-4, lr=1e-3){
  g <- build_generator(latent_dim)
  d_pack <- build_discriminator(dropout=dropout, l2=l2, lr=2e-4)  # 判别器学习率通常偏小更稳
  d <- d_pack$d; feat_mdl <- d_pack$feat
  
  n <- dim(X)[1]; steps <- ceiling(n/batch_size)
  for (ep in seq_len(epochs)){
    for (s in seq_len(steps)){
      idx <- sample.int(n, size = min(batch_size, n))
      ssgan_step(g, d_pack, feat_mdl,
                 batch_real_x = X[idx,,, , drop=FALSE],
                 batch_real_y = y[idx],
                 batch_size = batch_size, latent_dim = latent_dim)
    }
  }
  list(g=g, d=d, feat=feat_mdl)
}

disc_prob <- function(d, X, y=NULL){
  y_if <- if (is.null(y)) array(0, dim=c(dim(X)[1],1)) else array(y, dim=c(length(y),1))
  p <- predict(d, list(X, y_if))
  p[,2] / (p[,1] + p[,2] + 1e-8)
}

folds <- createFolds(ytr, k = 5, list = TRUE, returnTrain = FALSE)
grid <- expand.grid(dropout=c(0.2,0.3), l2=c(1e-4,3e-4), lr=c(1e-3,5e-4), stringsAsFactors=FALSE)
grid$mean_auc <- NA_real_

for (i in seq_len(nrow(grid))){
  dp <- grid$dropout[i]; l2v <- grid$l2[i]; lrv <- grid$lr[i]
  aucs <- numeric(length(folds))
  for (k in seq_along(folds)){
    va <- folds[[k]]; tr <- setdiff(seq_len(length(ytr)), va)
    mdl <- train_cssgan(Xtr[tr,,, ,drop=FALSE], ytr[tr],
                        epochs=60, batch_size=64,
                        dropout=dp, l2=l2v, lr=lrv)
    pr <- disc_prob(mdl$d, Xtr[va,,, ,drop=FALSE], y = ytr[va])
    aucs[k] <- as.numeric(pROC::auc(pROC::roc(ytr[va], pr, quiet=TRUE)))
    k_clear_session()
  }
  grid$mean_auc[i] <- mean(aucs, na.rm = TRUE)
}

grid <- grid[order(-grid$mean_auc, grid$dropout, grid$l2, grid$lr), ]
best <- grid[1, , drop=FALSE]
print(grid)
cat(sprintf("Best params -> dropout=%.2f, L2=%.4g, lr=%.4g | mean CV AUC=%.4f\n",
            best$dropout, best$l2, best$lr, best$mean_auc))

mdl_full <- train_cssgan(Xtr, ytr, epochs=60, batch_size=64,
                         dropout=best$dropout, l2=best$l2, lr=best$lr)

pr_tr <- disc_prob(mdl_full$d, Xtr, y = ytr)
pr_te <- disc_prob(mdl_full$d, Xte, y = yte)

auc_tr <- as.numeric(pROC::auc(pROC::roc(ytr, pr_tr, quiet=TRUE)))
auc_te <- as.numeric(pROC::auc(pROC::roc(yte,  pr_te, quiet=TRUE)))
cat(sprintf("Final Train AUC: %.4f\n", auc_tr))
cat(sprintf("Final Test  AUC: %.4f\n",  auc_te))

train_probe <- data.frame(ID = if ("ID" %in% names(train)) train$ID else sprintf("train_%d", seq_len(nrow(train))),
                          SSGAN_FM = pr_tr)
test_probe  <- data.frame(ID = if ("ID" %in% names(test))  test$ID  else sprintf("test_%d",  seq_len(nrow(test))),
                          SSGAN_FM = pr_te)
