############################################################
# Лабораторна робота №2
# CNN у R (keras/tensorflow) + Індивідуальне завдання №9
# Аугментація: flip vs flip+rotation vs full set
# Датасет: Fashion-MNIST
# Сумісно з Keras 3 / TF 2.20
############################################################

# --- 0) Пакети --------------------------------------------------------------
req <- c("reticulate","keras","tensorflow","ggplot2","dplyr","yardstick","tibble")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(reticulate)
library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(yardstick)
library(tibble)

# --- 0.1) SSL fix для завантаження датасету (якщо треба) ---------------------
py_install("certifi", pip = TRUE)
ca_path <- tryCatch(py_eval("(__import__('certifi')).where()"), error = function(e) NA)
if (!is.na(ca_path) && is.character(ca_path) && length(ca_path) == 1 && file.exists(ca_path)) {
  Sys.setenv(SSL_CERT_FILE = ca_path)
  Sys.setenv(REQUESTS_CA_BUNDLE = ca_path)
  Sys.setenv(CURL_CA_BUNDLE = ca_path)
}

# --- 0.2) Відтворюваність ---------------------------------------------------
set.seed(42)
tf$random$set_seed(42L)

cat("R:", R.version.string, "\n")
cat("TensorFlow:", tf$version$VERSION, "\n")
cat("Keras (R package):", as.character(packageVersion("keras")), "\n\n")

# --- 1) Дані: Fashion-MNIST -------------------------------------------------
mnist <- dataset_fashion_mnist()
c(x_all, y_all) %<-% mnist$train
c(x_test, y_test) %<-% mnist$test

# нормалізація
x_all  <- x_all / 255
x_test <- x_test / 255

# переконаємося що dtype правильний
x_all <- array(as.numeric(x_all), dim = dim(x_all))
x_test <- array(as.numeric(x_test), dim = dim(x_test))

# додати канал
x_all  <- array_reshape(x_all,  c(nrow(x_all),  28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# !!! важливо: мітки як integer
y_all  <- as.integer(y_all)
y_test <- as.integer(y_test)

input_shape <- c(28,28,1)
num_classes <- 10L

class_names <- c(
  "T-shirt/top","Trouser","Pullover","Dress","Coat",
  "Sandal","Shirt","Sneaker","Bag","Ankle boot"
)

# --- 1.1) Явний train/val split ----------------------------------------------
n_all <- dim(x_all)[1]
val_frac <- 0.2
n_val <- as.integer(round(n_all * val_frac))
n_train <- n_all - n_val

idx <- sample.int(n_all)
idx_train <- idx[1:n_train]
idx_val   <- idx[(n_train + 1):n_all]

x_train <- x_all[idx_train,,, , drop = FALSE]
y_train <- y_all[idx_train]
x_val   <- x_all[idx_val,,, , drop = FALSE]
y_val   <- y_all[idx_val]

cat(sprintf("Split: train=%d, val=%d, test=%d\n", length(y_train), length(y_val), length(y_test)))

# Конвертація в TensorFlow tensors
x_train_tf <- tf$constant(x_train, dtype = tf$float32)
y_train_tf <- tf$constant(y_train, dtype = tf$int64)
x_val_tf <- tf$constant(x_val, dtype = tf$float32)
y_val_tf <- tf$constant(y_val, dtype = tf$int64)
x_test_tf <- tf$constant(x_test, dtype = tf$float32)
y_test_tf <- tf$constant(y_test, dtype = tf$int64)

# --- 2) Аугментація (preprocessing layers) ----------------------------------
build_aug_layers <- function(mode = c("flip","flip_rot","full")) {
  mode <- match.arg(mode)
  
  if (mode == "flip") {
    return(list(layer_random_flip(mode = "horizontal")))
  }
  if (mode == "flip_rot") {
    return(list(
      layer_random_flip(mode = "horizontal"),
      layer_random_rotation(factor = 0.1)
    ))
  }
  # full
  list(
    layer_random_flip(mode = "horizontal"),
    layer_random_rotation(factor = 0.1),
    layer_random_translation(height_factor = 0.1, width_factor = 0.1),
    layer_random_zoom(height_factor = 0.1, width_factor = 0.1)
  )
}

# --- 3) CNN (Functional API) -------------------------------------------------
build_cnn <- function(aug_mode = c("flip","flip_rot","full"),
                      input_shape = c(28,28,1),
                      num_classes = 10L,
                      lr = 1e-3) {
  aug_mode <- match.arg(aug_mode)
  
  inputs <- layer_input(shape = input_shape, name = "input")
  x <- inputs
  
  aug_layers <- build_aug_layers(aug_mode)
  if (length(aug_layers) > 0) {
    for (L in aug_layers) x <- L(x)
  }
  
  x <- x |>
    layer_conv_2d(32, 3, padding = "same", activation = "relu") |>
    layer_batch_normalization() |>
    layer_conv_2d(32, 3, activation = "relu") |>
    layer_max_pooling_2d(2) |>
    layer_dropout(0.25) |>
    layer_conv_2d(64, 3, padding = "same", activation = "relu") |>
    layer_batch_normalization() |>
    layer_conv_2d(64, 3, activation = "relu") |>
    layer_max_pooling_2d(2) |>
    layer_dropout(0.25) |>
    layer_flatten() |>
    layer_dense(128, activation = "relu") |>
    layer_dropout(0.5)
  
  outputs <- x |> layer_dense(num_classes, activation = "softmax", name = "pred")
  model <- keras_model(inputs, outputs, name = paste0("cnn_", aug_mode))
  
  model$compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = list("accuracy")
  )
  model
}

# --- 4) MLP baseline ---------------------------------------------------------
build_mlp <- function(input_shape = c(28,28,1), num_classes = 10L, lr = 1e-3) {
  inputs <- layer_input(shape = input_shape)
  x <- inputs |>
    layer_flatten() |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(256, activation = "relu") |>
    layer_dropout(0.5)
  outputs <- x |> layer_dense(num_classes, activation = "softmax")
  
  model <- keras_model(inputs, outputs, name = "mlp_baseline")
  model$compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = list("accuracy")
  )
  model
}

# --- 5) Callbacks + helpers --------------------------------------------------
make_callbacks <- function() {
  list(
    callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 2, min_lr = 1e-6)
  )
}

history_to_df <- function(history, tag) {
  tibble(
    epoch = seq_along(history$history$loss),
    loss = history$history$loss,
    val_loss = history$history$val_loss,
    acc = history$history$accuracy,
    val_acc = history$history$val_accuracy,
    model = tag
  )
}

# --- 6) НАВЧАННЯ CNN: індивідуальне №9 (3 сценарії) --------------------------
# Вимога: flip vs flip+rotation vs full set
cat("\n===============================================\n")
cat("ІНДИВІДУАЛЬНЕ ЗАВДАННЯ №9\n")
cat("Порівняння 3 сценаріїв аугментації:\n")
cat("1. Тільки flip (horizontal)\n")
cat("2. Flip + rotation\n")
cat("3. Повний набір (flip + rotation + translation + zoom)\n")
cat("===============================================\n\n")

aug_scenarios <- c("flip","flip_rot","full")

epochs <- 5
batch_size <- 128L
cb <- make_callbacks()

histories <- list()
models <- list()
summary_rows <- list()

for (mode in aug_scenarios) {
  cat("\n=============================\n")
  cat("Training CNN with aug_mode:", mode, "\n")
  cat("=============================\n")
  
  model_cnn <- build_cnn(mode, input_shape, num_classes, lr = 1e-3)
  
  history <- model_cnn$fit(
    x = x_train_tf, 
    y = y_train_tf,
    validation_data = list(x_val_tf, y_val_tf),
    epochs = as.integer(epochs),
    batch_size = as.integer(batch_size),
    callbacks = cb,
    verbose = 2
  )
  
  models[[mode]] <- model_cnn
  histories[[mode]] <- history
  
  dfh <- history_to_df(history, paste0("cnn_", mode))
  
  summary_rows[[mode]] <- dfh |>
    summarize(
      aug_mode = mode,
      best_val_acc  = max(val_acc, na.rm = TRUE),
      best_val_loss = min(val_loss, na.rm = TRUE),
      final_val_acc  = dplyr::last(val_acc),
      final_val_loss = dplyr::last(val_loss),
      epochs_trained = n()
    )
}

summary_tbl <- bind_rows(summary_rows) |> arrange(desc(best_val_acc))

cat("\n\n===============================================\n")
cat("РЕЗУЛЬТАТИ ІНДИВІДУАЛЬНОГО ЗАВДАННЯ №9\n")
cat("Порівняння 3 сценаріїв аугментації\n")
cat("===============================================\n")
print(summary_tbl)

# Аналіз результатів
cat("\n--- АНАЛІЗ ---\n")
best_aug <- summary_tbl$aug_mode[1]
cat(sprintf("Найкраща аугментація: %s (val_acc = %.4f)\n", 
            best_aug, summary_tbl$best_val_acc[1]))

cat("\nПорівняння best_val_acc:\n")
for(i in 1:nrow(summary_tbl)) {
  cat(sprintf("  %d. %s: %.4f\n", i, summary_tbl$aug_mode[i], summary_tbl$best_val_acc[i]))
}

# --- 7) Візуалізація (за вимогою) -------------------------------------------
df_hist_all <- bind_rows(lapply(names(histories), function(mode) {
  history_to_df(histories[[mode]], tag = paste0("cnn_", mode))
}))

# (A) порівняння val_acc - ГОЛОВНИЙ ГРАФІК для завдання
p_val_acc <- ggplot(df_hist_all, aes(epoch, val_acc, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2, alpha = 0.6) +
  labs(title = "Індивідуальне завдання №9: Validation Accuracy",
       subtitle = "Порівняння: flip vs flip+rotation vs full augmentation",
       x = "Epoch", y = "Validation Accuracy",
       color = "Augmentation") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 11))
print(p_val_acc)

# (B) порівняння val_loss
p_val_loss <- ggplot(df_hist_all, aes(epoch, val_loss, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2, alpha = 0.6) +
  labs(title = "Індивідуальне завдання №9: Validation Loss",
       subtitle = "Порівняння: flip vs flip+rotation vs full augmentation",
       x = "Epoch", y = "Validation Loss",
       color = "Augmentation") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 11))
print(p_val_loss)

# (C) фасети для звіту
print(
  ggplot(df_hist_all, aes(epoch, val_acc)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_point(color = "steelblue", size = 1.5, alpha = 0.6) +
    facet_wrap(~ model, ncol = 3) +
    labs(title = "Val Accuracy по сценаріях (фасети)", 
         y = "Val Accuracy", x = "Epoch") +
    theme_minimal()
)

print(
  ggplot(df_hist_all, aes(epoch, val_loss)) +
    geom_line(color = "coral", linewidth = 1) +
    geom_point(color = "coral", size = 1.5, alpha = 0.6) +
    facet_wrap(~ model, ncol = 3) +
    labs(title = "Val Loss по сценаріях (фасети)", 
         y = "Val Loss", x = "Epoch") +
    theme_minimal()
)

# (D) Барплот порівняння best_val_acc
bar_data <- summary_tbl |> 
  mutate(aug_mode = factor(aug_mode, levels = c("flip", "flip_rot", "full")))

p_bar <- ggplot(bar_data, aes(x = aug_mode, y = best_val_acc, fill = aug_mode)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.4f", best_val_acc)), 
            vjust = -0.5, size = 4) +
  labs(title = "Порівняння найкращої Validation Accuracy",
       subtitle = "Індивідуальне завдання №9",
       x = "Тип аугментації", y = "Best Validation Accuracy") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))
print(p_bar)

# --- 8) Найкраща модель і тестова оцінка ------------------------------------
best_mode <- summary_tbl$aug_mode[1]
cat("\n\n===============================================\n")
cat("ТЕСТУВАННЯ НАЙКРАЩОЇ МОДЕЛІ\n")
cat("===============================================\n")
cat("Best augmentation mode:", best_mode, "\n")
best_cnn <- models[[best_mode]]

cat("\n--- Test evaluation (best CNN) ---\n")
best_eval <- best_cnn$evaluate(x_test_tf, y_test_tf, verbose = 0)
print(best_eval)

pred_prob <- best_cnn$predict(x_test_tf, verbose = 0)
pred_cls <- max.col(pred_prob) - 1L

conf <- table(
  Predicted = factor(pred_cls, levels = 0:9, labels = class_names),
  Actual    = factor(y_test,  levels = 0:9, labels = class_names)
)
cat("\n--- Confusion matrix (best CNN) ---\n")
print(conf)

conf_df <- as.data.frame(conf) |> rename(n = Freq)
print(
  ggplot(conf_df, aes(x = Actual, y = Predicted, fill = n)) +
    geom_tile(color = "white") +
    geom_text(aes(label = n), color = "white", size = 3) +
    scale_fill_gradient(low = "steelblue", high = "darkred") +
    labs(title = paste("Confusion Matrix - Best CNN:", best_mode),
         x = "Actual Class", y = "Predicted Class") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
)

df_pred <- tibble(
  truth = factor(y_test, levels = 0:9, labels = class_names),
  estimate = factor(pred_cls, levels = 0:9, labels = class_names)
)

cat("\nAccuracy (best CNN):\n")
acc_result <- accuracy(df_pred, truth = truth, estimate = estimate)
print(acc_result)

cat("\nF1 macro (best CNN):\n")
f1_result <- f_meas(df_pred, truth = truth, estimate = estimate, estimator = "macro")
print(f1_result)

# --- 9) CNN vs MLP -----------------------------------------------------------
cat("\n=============================\n")
cat("Training MLP baseline\n")
cat("=============================\n")

model_mlp <- build_mlp(input_shape, num_classes, lr = 1e-3)
hist_mlp <- model_mlp$fit(
  x = x_train_tf,
  y = y_train_tf,
  validation_data = list(x_val_tf, y_val_tf),
  epochs = as.integer(5),
  batch_size = as.integer(batch_size),
  callbacks = cb,
  verbose = 2
)

cat("\n--- Test evaluation (MLP) ---\n")
mlp_eval <- model_mlp$evaluate(x_test_tf, y_test_tf, verbose = 0)
print(mlp_eval)

compare_tbl <- tibble(
  model = c(paste0("CNN_best(", best_mode, ")"), "MLP"),
  test_loss = c(as.numeric(best_eval[[1]]), as.numeric(mlp_eval[[1]])),  # перший елемент - loss
  test_acc  = c(as.numeric(best_eval[[2]]), as.numeric(mlp_eval[[2]]))   # другий елемент - accuracy
)
cat("\n--- CNN vs MLP comparison ---\n")
print(compare_tbl)

# Графік порівняння CNN vs MLP
p_compare <- ggplot(compare_tbl, aes(x = model, y = test_acc, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = sprintf("%.4f", test_acc)), vjust = -0.5, size = 4) +
  labs(title = "CNN (з найкращою аугментацією) vs MLP",
       x = "Model", y = "Test Accuracy") +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  theme(legend.position = "none")
print(p_compare)

# --- 10) Збереження / відновлення -------------------------------------------
dir.create("models", showWarnings = FALSE, recursive = TRUE)
cnn_save_path <- file.path("models", paste0("cnn_fashion_mnist_best_", best_mode, ".keras"))

cat("\nSaving best CNN to:", cnn_save_path, "\n")
best_cnn$save(cnn_save_path)

cat("Loading model back...\n")
restored <- keras$models$load_model(cnn_save_path)

cat("\n--- Test evaluation (restored CNN) ---\n")
restored_eval <- restored$evaluate(x_test_tf, y_test_tf, verbose = 0)
print(restored_eval)

# --- 11) Відтворюваність -----------------------------------------------------
cat("\n\n===============================================\n")
cat("REPRODUCIBILITY INFO\n")
cat("===============================================\n")
cat("R version:", R.version.string, "\n")
cat("TensorFlow version:", tf$version$VERSION, "\n")
cat("Seeds used: set.seed(42), tf$random$set_seed(42L)\n\n")

print(as.data.frame(installed.packages()[c("keras","tensorflow","ggplot2","dplyr","yardstick"),
                                         c("Package","Version")]))

# --- 12) ФІНАЛЬНИЙ ВИСНОВОК --------------------------------------------------
cat("\n\n===============================================\n")
cat("ВИСНОВКИ ПО ІНДИВІДУАЛЬНОМУ ЗАВДАННЮ №9\n")
cat("===============================================\n\n")

cat("Порівняння 3 сценаріїв аугментації:\n\n")

for(i in 1:nrow(summary_tbl)) {
  mode_name <- summary_tbl$aug_mode[i]
  mode_desc <- switch(mode_name,
                      "flip" = "Тільки горизонтальний flip",
                      "flip_rot" = "Flip + обертання (±10°)",
                      "full" = "Повний набір (flip + rotation + translation + zoom)")
  cat(sprintf("%d. %s (%s)\n", i, mode_name, mode_desc))
  cat(sprintf("   Best val_acc: %.4f\n", summary_tbl$best_val_acc[i]))
  cat(sprintf("   Best val_loss: %.4f\n", summary_tbl$best_val_loss[i]))
  cat(sprintf("   Epochs trained: %d\n\n", summary_tbl$epochs_trained[i]))
}

cat("Узагальнення:\n")
cat(sprintf("- Найкраща аугментація: %s\n", best_aug))
cat(sprintf("- Test accuracy найкращої моделі: %.4f\n", as.numeric(best_eval[[2]])))
cat(sprintf("- Перевага CNN над MLP: %.4f\n", 
            as.numeric(best_eval[[2]]) - as.numeric(mlp_eval[[2]])))

cat("\nАугментація даних покращує узагальнення моделі та зменшує перенавчання.\n")
cat("Складніші методи аугментації можуть як покращити, так і погіршити результати\n")
cat("залежно від специфіки датасету.\n")
