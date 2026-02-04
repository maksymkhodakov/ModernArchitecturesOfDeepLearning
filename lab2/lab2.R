############################################################
# Лабораторна робота №2 (повний робочий скрипт)
# Тема: CNN у R (keras/tensorflow) + Індивідуальне завдання №9
# Аугментація: none vs flip vs flip+rotation vs full set
#
# Датасет: Fashion-MNIST (28x28x1, 10 класів)
#
# Функціонал:
# 1) Підготовка середовища + відтворюваність (seed)
# 2) Завантаження/нормалізація даних
# 3) CNN з Conv/Pool/BN/Dropout
# 4) Колбеки EarlyStopping + ReduceLROnPlateau
# 5) Навчання CNN у 4 сценаріях аугментації (індивідуальне завдання)
# 6) Графіки train/val accuracy та loss
# 7) Оцінка на тесті, confusion matrix + yardstick метрики
# 8) Порівняння з MLP
# 9) Збереження/відновлення найкращої CNN (SavedModel)
# 10) Вивід версій пакетів і TF для відтворюваності
############################################################

# --- 0) Підготовка середовища ------------------------------------------------
req <- c("keras", "tensorflow", "ggplot2", "dplyr", "yardstick", "tibble")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(yardstick)
library(tibble)

# (перший раз) інсталяція бекенду TF — розкоментуй за потреби:
# tensorflow::install_tensorflow()

# Відтворюваність
set.seed(42)
tensorflow::tf$random$set_seed(42L)

cat("R:", R.version.string, "\n")
cat("TensorFlow:", tf$version$VERSION, "\n")
cat("Keras (R package):", as.character(packageVersion("keras")), "\n\n")

# --- 1) Завантаження і підготовка даних -------------------------------------
mnist <- dataset_fashion_mnist()
c(x_train, y_train) %<-% mnist$train
c(x_test, y_test) %<-% mnist$test

# Нормалізація [0,1]
x_train <- x_train / 255
x_test  <- x_test / 255

# Додати канал (N, 28, 28, 1)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test  <- array_reshape(x_test,  c(nrow(x_test),  28, 28, 1))

num_classes <- 10L
input_shape <- c(28, 28, 1)

# Назви класів
class_names <- c(
  "T-shirt/top","Trouser","Pullover","Dress","Coat",
  "Sandal","Shirt","Sneaker","Bag","Ankle boot"
)

# --- 2) Функції: аугментація, моделі, колбеки -------------------------------

# 2.1) Блок аугментації (preprocessing layers)
# mode: "none" | "flip" | "flip_rot" | "full"
build_augmentation <- function(mode = c("none", "flip", "flip_rot", "full")) {
  mode <- match.arg(mode)

  if (mode == "none") {
    return(NULL) # немає аугментації
  }
  if (mode == "flip") {
    return(
      keras_model_sequential(name = "aug_flip") |>
        layer_random_flip(mode = "horizontal")
    )
  }
  if (mode == "flip_rot") {
    return(
      keras_model_sequential(name = "aug_flip_rot") |>
        layer_random_flip(mode = "horizontal") |>
        layer_random_rotation(factor = 0.1)
    )
  }

  # full
  keras_model_sequential(name = "aug_full") |>
    layer_random_flip(mode = "horizontal") |>
    layer_random_rotation(factor = 0.1) |>
    layer_random_translation(height_factor = 0.1, width_factor = 0.1) |>
    layer_random_zoom(height_factor = 0.1, width_factor = 0.1)
}

# 2.2) CNN модель (важливо: у R НЕ можна робити |> aug |> ...)
# тому аугментацію додаємо через layer(aug) або без неї
build_cnn <- function(aug_mode = c("none", "flip", "flip_rot", "full"),
                      input_shape = c(28,28,1),
                      num_classes = 10L,
                      lr = 1e-3) {
  aug_mode <- match.arg(aug_mode)
  aug <- build_augmentation(aug_mode)

  model <- keras_model_sequential(name = paste0("cnn_", aug_mode))

  # Додаємо аугментацію як перший шар, якщо вона є
  if (!is.null(aug)) {
    model <- model |> layer(aug)
  }

  # Основна CNN-архітектура
  model <- model |>
    layer_conv_2d(filters = 32, kernel_size = 3, padding = "same",
                  activation = "relu", input_shape = input_shape) |>
    layer_batch_normalization() |>
    layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
    layer_max_pooling_2d(pool_size = 2) |>
    layer_dropout(0.25) |>

    layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                  activation = "relu") |>
    layer_batch_normalization() |>
    layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
    layer_max_pooling_2d(pool_size = 2) |>
    layer_dropout(0.25) |>

    layer_flatten() |>
    layer_dense(128, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(num_classes, activation = "softmax")

  model |>
    compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = optimizer_adam(learning_rate = lr),
      metrics = c("accuracy")
    )

  model
}

# 2.3) MLP (baseline для порівняння)
build_mlp <- function(input_shape = c(28,28,1), num_classes = 10L, lr = 1e-3) {
  model <- keras_model_sequential(name = "mlp_baseline") |>
    layer_flatten(input_shape = input_shape) |>
    layer_dense(512, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(256, activation = "relu") |>
    layer_dropout(0.5) |>
    layer_dense(num_classes, activation = "softmax")

  model |>
    compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = optimizer_adam(learning_rate = lr),
      metrics = c("accuracy")
    )

  model
}

# 2.4) Колбеки
make_callbacks <- function() {
  list(
    callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 2, min_lr = 1e-6)
  )
}

# 2.5) Перетворення history -> tibble
history_to_df <- function(history, tag = "model") {
  tibble(
    epoch = seq_along(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss,
    acc = history$metrics$accuracy,
    val_acc = history$metrics$val_accuracy,
    model = tag
  )
}

# --- 3) Навчання CNN: 4 сценарії аугментації (індивідуальне завдання №9) ----
aug_scenarios <- c("none", "flip", "flip_rot", "full")

epochs <- 30
batch_size <- 128
val_split <- 0.2
cb <- make_callbacks()

histories <- list()
models <- list()
summary_rows <- list()

for (mode in aug_scenarios) {
  cat("\n=============================\n")
  cat("Training CNN with aug_mode:", mode, "\n")
  cat("=============================\n")

  model_cnn <- build_cnn(
    aug_mode = mode,
    input_shape = input_shape,
    num_classes = num_classes,
    lr = 1e-3
  )

  history <- model_cnn |>
    fit(
      x = x_train, y = y_train,
      validation_split = val_split,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = cb,
      verbose = 2
    )

  models[[mode]] <- model_cnn
  histories[[mode]] <- history

  dfh <- history_to_df(history, tag = paste0("cnn_", mode))

  summary_rows[[mode]] <- dfh |>
    summarize(
      aug_mode = mode,
      best_val_acc = max(val_acc, na.rm = TRUE),
      best_val_loss = min(val_loss, na.rm = TRUE),
      final_val_acc = last(val_acc),
      final_val_loss = last(val_loss)
    )
}

summary_tbl <- bind_rows(summary_rows) |>
  arrange(desc(best_val_acc))

cat("\n--- Augmentation comparison (индив. завдання №9) ---\n")
print(summary_tbl)

# --- 4) Графіки навчання (всі сценарії) -------------------------------------
df_hist_all <- bind_rows(lapply(names(histories), function(mode) {
  history_to_df(histories[[mode]], tag = paste0("cnn_", mode))
}))

# Validation Accuracy (головне для порівняння аугментації)
ggplot(df_hist_all, aes(x = epoch, y = val_acc, color = model)) +
  geom_line() +
  labs(title = "CNN: Validation Accuracy (порівняння аугментацій)",
       x = "Epoch", y = "Val Accuracy") +
  theme(legend.position = "bottom")

# Validation Loss
ggplot(df_hist_all, aes(x = epoch, y = val_loss, color = model)) +
  geom_line() +
  labs(title = "CNN: Validation Loss (порівняння аугментацій)",
       x = "Epoch", y = "Val Loss") +
  theme(legend.position = "bottom")

# (Опційно) по фасетах — зручно для звіту
ggplot(df_hist_all, aes(epoch, val_acc)) +
  geom_line() +
  facet_wrap(~ model, ncol = 2) +
  labs(title = "CNN: Val Accuracy по сценаріях", y = "Val Accuracy")

ggplot(df_hist_all, aes(epoch, val_loss)) +
  geom_line() +
  facet_wrap(~ model, ncol = 2) +
  labs(title = "CNN: Val Loss по сценаріях", y = "Val Loss")

# --- 5) Вибір найкращого CNN за best_val_acc --------------------------------
best_mode <- summary_tbl$aug_mode[1]
cat("\nBest augmentation mode by best_val_acc:", best_mode, "\n")
best_cnn <- models[[best_mode]]

# --- 6) Оцінка на тесті + confusion matrix + yardstick метрики --------------
cat("\n--- Test evaluation (best CNN) ---\n")
best_eval <- best_cnn |> evaluate(x_test, y_test, verbose = 0)
print(best_eval)

pred_prob <- best_cnn |> predict(x_test, verbose = 0)
pred_cls <- max.col(pred_prob) - 1L  # 0..9

conf <- table(
  Predicted = factor(pred_cls, levels = 0:9, labels = class_names),
  Actual    = factor(y_test,  levels = 0:9, labels = class_names)
)
cat("\n--- Confusion matrix (best CNN) ---\n")
print(conf)

# Heatmap confusion matrix (для рисунку у звіт)
conf_df <- as.data.frame(conf) |>
  rename(n = Freq)

ggplot(conf_df, aes(x = Actual, y = Predicted, fill = n)) +
  geom_tile() +
  labs(title = paste("Confusion Matrix (best CNN:", best_mode, ")"),
       x = "Actual", y = "Predicted") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Метрики
df_pred <- tibble(
  truth = factor(y_test, levels = 0:9, labels = class_names),
  estimate = factor(pred_cls, levels = 0:9, labels = class_names)
)

cat("\nAccuracy (best CNN):\n")
print(accuracy(df_pred, truth = truth, estimate = estimate))

cat("\nF1 macro (best CNN):\n")
print(f_meas(df_pred, truth = truth, estimate = estimate, estimator = "macro"))

# --- 7) Порівняння CNN vs MLP -----------------------------------------------
cat("\n=============================\n")
cat("Training MLP baseline (no augmentation)\n")
cat("=============================\n")

model_mlp <- build_mlp(input_shape = input_shape, num_classes = num_classes, lr = 1e-3)
cb_mlp <- make_callbacks()

hist_mlp <- model_mlp |>
  fit(
    x = x_train, y = y_train,
    validation_split = val_split,
    epochs = 20,
    batch_size = batch_size,
    callbacks = cb_mlp,
    verbose = 2
  )

cat("\n--- Test evaluation (MLP) ---\n")
mlp_eval <- model_mlp |> evaluate(x_test, y_test, verbose = 0)
print(mlp_eval)

compare_tbl <- tibble(
  model = c(paste0("CNN_best(", best_mode, ")"), "MLP"),
  test_loss = c(as.numeric(best_eval["loss"]), as.numeric(mlp_eval["loss"])),
  test_acc  = c(as.numeric(best_eval["accuracy"]), as.numeric(mlp_eval["accuracy"]))
)
cat("\n--- CNN vs MLP comparison ---\n")
print(compare_tbl)

# --- 8) Збереження / відновлення найкращої CNN ------------------------------
dir.create("models", showWarnings = FALSE, recursive = TRUE)
cnn_save_path <- file.path("models", paste0("cnn_fashion_mnist_best_", best_mode))

cat("\nSaving best CNN to:", cnn_save_path, "\n")
save_model_tf(best_cnn, cnn_save_path)

cat("Loading model back...\n")
restored <- load_model_tf(cnn_save_path)

cat("\n--- Test evaluation (restored CNN) ---\n")
restored_eval <- restored |> evaluate(x_test, y_test, verbose = 0)
print(restored_eval)

# --- 9) Відтворюваність: версії пакетів -------------------------------------
cat("\n--- Reproducibility info ---\n")
cat("R version:\n")
print(R.version.string)

cat("\nPackage versions:\n")
print(as.data.frame(installed.packages()[c("keras","tensorflow","ggplot2","dplyr","yardstick"), c("Package","Version")]))

cat("\nSeeds used: set.seed(42), tf$random$set_seed(42L)\n")
