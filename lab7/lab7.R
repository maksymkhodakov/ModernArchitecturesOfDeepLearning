############################################################
# Лабораторна 7: Невизначеність у DL (R/torch)
# MC-Dropout vs Deep Ensembles + Індивідуальне №9:
# Selective prediction: accuracy@coverage(τ)
############################################################

# ----------------------------------------------------------
# 0) Пакети R
# ----------------------------------------------------------
req <- c("torch","ggplot2","dplyr","tidyr","tibble","purrr","readr","forcats")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(torch)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(readr)
library(forcats)

# ----------------------------------------------------------
# 1) Папка для результатів
# ----------------------------------------------------------
out_dir <- "plots_uncertainty_lab7"
if (!dir.exists(out_dir)) dir.create(out_dir)

save_plot <- function(name, p, w=8, h=6, dpi=250) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, p, width=w, height=h, dpi=dpi, bg="white")
  cat("  ✓ Збережено:", path, "\n")
}

# ----------------------------------------------------------
# 2) Фіксація seed (відтворюваність)
# ----------------------------------------------------------
set.seed(42)
torch_manual_seed(42)

# ----------------------------------------------------------
# 3) Дані: Iris (3 класи)
# ----------------------------------------------------------
data(iris)

X <- as.matrix(scale(iris[, 1:4]))
y <- as.integer(iris$Species)  # 1..3
n <- nrow(X)

# Train/Val/Test split (щоб було "чесно" для порогу τ)
# 60/20/20
idx_all <- sample(seq_len(n))
n_tr <- floor(0.6*n)
n_val <- floor(0.2*n)

idx_tr <- idx_all[1:n_tr]
idx_val <- idx_all[(n_tr+1):(n_tr+n_val)]
idx_te <- idx_all[(n_tr+n_val+1):n]

X_tr <- X[idx_tr,]; y_tr <- y[idx_tr]
X_val <- X[idx_val,]; y_val <- y[idx_val]
X_te <- X[idx_te,]; y_te <- y[idx_te]

# Torch tensors (залишаємо 1-based indexing для torch R)
x_tr <- torch_tensor(X_tr, dtype = torch_float())
y_tr <- torch_tensor(y_tr, dtype = torch_long())  # 1..3
x_val <- torch_tensor(X_val, dtype = torch_float())
y_val <- torch_tensor(y_val, dtype = torch_long())
x_te <- torch_tensor(X_te, dtype = torch_float())
y_te <- torch_tensor(y_te, dtype = torch_long())

cat("Shapes:",
    "train =", nrow(X_tr),
    "| val =", nrow(X_val),
    "| test =", nrow(X_te), "\n")

# ----------------------------------------------------------
# 4) Модель (без BatchNorm!) з Dropout
# ----------------------------------------------------------
net_cls <- nn_module(
  initialize = function(in_dim=4, hid=32, out_dim=3, p_drop=0.5) {
    self$fc1 <- nn_linear(in_dim, hid)
    self$do1 <- nn_dropout(p_drop)
    self$fc2 <- nn_linear(hid, hid)
    self$do2 <- nn_dropout(p_drop)
    self$out <- nn_linear(hid, out_dim)
  },
  forward = function(x) {
    x %>%
      self$fc1() %>% nnf_relu() %>% self$do1() %>%
      self$fc2() %>% nnf_relu() %>% self$do2() %>%
      self$out()
  }
)

# ----------------------------------------------------------
# 5) Тренування 1 моделі
# ----------------------------------------------------------
train_one <- function(epochs=400, lr=5e-3, p_drop=0.5, weight_decay=0) {
  model <- net_cls(p_drop = p_drop)
  opt <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)

  for (e in 1:epochs) {
    model$train()
    opt$zero_grad()
    logits <- model(x_tr)
    loss <- nnf_cross_entropy(logits, y_tr)
    loss$backward()
    opt$step()
  }

  model$eval()
  model
}

# ----------------------------------------------------------
# 6) Хелпери: softmax, ентропія, accuracy
# ----------------------------------------------------------
softmax_np <- function(logits) {
  # logits: torch tensor [N, C]
  as_array(nnf_softmax(logits, dim = 2))
}

entropy_vec <- function(p_mat) {
  # p_mat: [N, C] probabilities
  apply(p_mat, 1, function(p) -sum(p * log(p + 1e-12)))
}

acc_from_probs <- function(p_mat, y_true0) {
  y_hat <- max.col(p_mat)  # max.col повертає 1-based індекси
  mean(y_hat == y_true0)
}

# ----------------------------------------------------------
# 7) MC-Dropout прогноз (T стохастичних проходів)
# ----------------------------------------------------------
mc_predict <- function(model, x, T=50L) {
  with_no_grad({
    N <- x$size(1)
    C <- 3
    probs_arr <- array(0, dim = c(N, C, T))

    for (t in 1:T) {
      model$train()  # важливо: щоб dropout був активним
      logits <- model(x)
      probs_arr[,,t] <- softmax_np(logits)
    }
    probs_arr
  })
}

# ----------------------------------------------------------
# 8) Deep Ensemble прогноз (M незалежних моделей)
# ----------------------------------------------------------
ensemble_predict <- function(models, x) {
  with_no_grad({
    p_sum <- NULL
    for (m in seq_along(models)) {
      models[[m]]$eval()
      logits <- models[[m]](x)
      p <- softmax_np(logits)
      if (is.null(p_sum)) p_sum <- p else p_sum <- p_sum + p
    }
    p_sum / length(models)
  })
}

# ----------------------------------------------------------
# 9) Навчання моделей: 1x MC-Dropout + Mx Ensemble
# ----------------------------------------------------------
cat("\n=== Тренування MC-Dropout моделі ===\n")
m_mc <- train_one(epochs=500, lr=5e-3, p_drop=0.5, weight_decay=1e-4)

T_mc <- 80L
cat("MC passes T =", T_mc, "\n")

# Deep ensemble
M <- 5
cat("\n=== Тренування Deep Ensemble: M =", M, "===\n")
mods <- vector("list", M)
for (i in 1:M) {
  torch_manual_seed(100 + i)      # різні ініціалізації
  set.seed(100 + i)
  mods[[i]] <- train_one(epochs=400, lr=8e-3, p_drop=0.5, weight_decay=1e-4)
  cat("  ✓ model", i, "готовий\n")
}

# ----------------------------------------------------------
# 10) Прогнози на VAL і TEST + невизначеність
# ----------------------------------------------------------
# --- MC on VAL
mc_val <- mc_predict(m_mc, x_val, T = T_mc)
p_mean_val_mc <- apply(mc_val, c(1,2), mean)
ent_val_mc <- entropy_vec(p_mean_val_mc)
conf_val_mc <- apply(p_mean_val_mc, 1, max)  # max-softmax prob

# --- MC on TEST
mc_te <- mc_predict(m_mc, x_te, T = T_mc)
p_mean_te_mc <- apply(mc_te, c(1,2), mean)
ent_te_mc <- entropy_vec(p_mean_te_mc)
conf_te_mc <- apply(p_mean_te_mc, 1, max)

acc_te_mc_full <- acc_from_probs(p_mean_te_mc, as.integer(as_array(y_te)))

# --- Ensemble on VAL/TEST
p_val_ens <- ensemble_predict(mods, x_val)
ent_val_ens <- entropy_vec(p_val_ens)
conf_val_ens <- apply(p_val_ens, 1, max)

p_te_ens <- ensemble_predict(mods, x_te)
ent_te_ens <- entropy_vec(p_te_ens)
conf_te_ens <- apply(p_te_ens, 1, max)

acc_te_ens_full <- acc_from_probs(p_te_ens, as.integer(as_array(y_te)))

cat(sprintf("\nFull coverage accuracy:\n  MC-Dropout=%.3f | Ensemble=%.3f\n",
            acc_te_mc_full, acc_te_ens_full))

# ----------------------------------------------------------
# 11) ІНДИВІДУАЛЬНЕ №9: Selective prediction
#     accuracy@coverage(τ) по порогу довіри τ = max prob
#     τ підбираємо на VAL, а оцінюємо на TEST (без leakage)
# ----------------------------------------------------------
accuracy_coverage_curve <- function(conf_val, conf_te, probs_te, y_te0,
                                    taus = seq(0.0, 0.99, by=0.01)) {
  y_te0 <- as.integer(y_te0)

  out <- lapply(taus, function(tau) {
    keep <- conf_te >= tau
    coverage <- mean(keep)

    if (sum(keep) == 0) {
      return(tibble(tau=tau, coverage=0, accuracy=NA_real_, n_kept=0))
    }
    p_keep <- probs_te[keep, , drop=FALSE]
    y_keep <- y_te0[keep]
    acc <- acc_from_probs(p_keep, y_keep)

    tibble(tau=tau, coverage=coverage, accuracy=acc, n_kept=sum(keep))
  })

  bind_rows(out)
}

y_te0 <- as.integer(as_array(y_te))

curve_mc  <- accuracy_coverage_curve(conf_val_mc,  conf_te_mc,  p_mean_te_mc, y_te0) %>%
  mutate(Method="MC-Dropout")
curve_ens <- accuracy_coverage_curve(conf_val_ens, conf_te_ens, p_te_ens,      y_te0) %>%
  mutate(Method="Deep Ensemble")

curve_all <- bind_rows(curve_mc, curve_ens)

# Зберігаємо таблицю
write_csv(curve_all, file.path(out_dir, "accuracy_coverage_table.csv"))
cat("  ✓ Збережено:", file.path(out_dir, "accuracy_coverage_table.csv"), "\n")

# ----------------------------------------------------------
# 12) Візуалізація
# ----------------------------------------------------------

# 12.1 Accuracy@Coverage
p_acc_cov <- ggplot(curve_all, aes(coverage, accuracy, color=Method)) +
  geom_line(linewidth=1) +
  geom_point(size=1.5, alpha=0.8) +
  scale_x_continuous(limits=c(0,1)) +
  labs(
    title="Selective prediction: Accuracy@Coverage(τ)",
    x="Coverage (частка прикритих прикладів)",
    y="Accuracy на прикритих прикладах",
    color="Метод"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_01_accuracy_at_coverage", p_acc_cov, w=8, h=6)

# 12.2 Coverage@Tau
p_cov_tau <- ggplot(curve_all, aes(tau, coverage, color=Method)) +
  geom_line(linewidth=1) +
  labs(
    title="Selective prediction: Coverage(τ)",
    x="Поріг довіри τ (max softmax probability)",
    y="Coverage",
    color="Метод"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_02_coverage_vs_tau", p_cov_tau)

# 12.3 Розподіл ентропій
df_ent <- bind_rows(
  tibble(entropy = ent_te_mc,  Method="MC-Dropout"),
  tibble(entropy = ent_te_ens, Method="Deep Ensemble")
)

p_ent <- ggplot(df_ent, aes(entropy, fill=Method)) +
  geom_histogram(bins=20, alpha=0.6, position="identity") +
  labs(
    title="Розподіл невизначеності (ентропія) на TEST",
    x="Ентропія (вище = менш впевнено)",
    y="Кількість"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_03_entropy_hist_test", p_ent)

# 12.4 Калібрування (reliability plot) для Ensemble і MC
# Реалізація: бінінг по confidence -> accuracy vs mean confidence
reliability_df <- function(conf, probs, y_true0, bins=10, name="") {
  y_true0 <- as.integer(y_true0)
  y_hat <- max.col(probs)
  correct <- (y_hat == y_true0)

  df <- tibble(conf=conf, correct=correct)
  df <- df %>%
    mutate(bin = cut(conf, breaks = seq(0,1,length.out=bins+1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarise(
      mean_conf = mean(conf),
      acc = mean(correct),
      n = n(),
      .groups="drop"
    ) %>%
    mutate(Method=name)
  df
}

rel_mc  <- reliability_df(conf_te_mc,  p_mean_te_mc, y_te0, bins=10, name="MC-Dropout")
rel_ens <- reliability_df(conf_te_ens, p_te_ens,     y_te0, bins=10, name="Deep Ensemble")
rel_all <- bind_rows(rel_mc, rel_ens)

p_rel <- ggplot(rel_all, aes(mean_conf, acc, color=Method)) +
  geom_line(linewidth=1) +
  geom_point(aes(size=n), alpha=0.8) +
  geom_abline(linetype=2) +
  scale_x_continuous(limits=c(0,1)) +
  scale_y_continuous(limits=c(0,1)) +
  labs(
    title="Reliability plot (калібрування) на TEST",
    x="Середня впевненість у біні",
    y="Точність у біні",
    color="Метод",
    size="n"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_04_reliability_plot", p_rel, w=8, h=6)

# ----------------------------------------------------------
# 13) Підсумок
# ----------------------------------------------------------
cat("\n=== ГОТОВО ===\n")
cat("Папка:", out_dir, "\n")
print(list.files(out_dir))
