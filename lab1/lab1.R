############################################################
# Лабораторна робота №1 — Варіант 9
# Простий Transformer Encoder у R (torch), без keras/tensorflow
############################################################

# --- 0) Підготовка -----------------------------------------------------------
if (!require(torch)) install.packages("torch")
library(torch)

set.seed(123)
torch_manual_seed(123)

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n")

# --- 1) Генерація штучних даних (синус + шум) -------------------------------
# Кожен приклад:
# X = s[1:seq_len]
# y = s[seq_len+1]  (наступний крок)

make_sine_dataset <- function(n_samples = 2048,
                              seq_len = 40,
                              noise_sd = 0.05,
                              dt = 0.15) {
  t <- seq(0, seq_len * dt, length.out = seq_len + 1)

  X <- matrix(0, nrow = n_samples, ncol = seq_len)
  y <- matrix(0, nrow = n_samples, ncol = 1)

  for (i in 1:n_samples) {
    amp   <- runif(1, 0.7, 1.3)
    freq  <- runif(1, 0.8, 1.4)
    phase <- runif(1, 0, 2*pi)

    s <- amp * sin(freq * t + phase) + rnorm(length(t), 0, noise_sd)

    X[i, ] <- s[1:seq_len]
    y[i, 1] <- s[seq_len + 1]
  }

  list(
    X = torch_tensor(X, dtype = torch_float()),
    y = torch_tensor(y, dtype = torch_float())
  )
}

seq_len <- 40
n_train <- 2048
n_test  <- 512

train_data <- make_sine_dataset(n_samples = n_train, seq_len = seq_len)
test_data  <- make_sine_dataset(n_samples = n_test,  seq_len = seq_len)

# --- 2) Positional Encoding (S, 1, E) ---------------------------------------
# Повертає pe форми (seq_len, 1, d_model), щоб додавати до (S, B, E)

positional_encoding <- function(seq_len, d_model, device = "cpu") {
  pe <- torch_zeros(c(seq_len, 1, d_model), device = device, dtype = torch_float())

  # position: (S, 1, 1)
  position <- torch_arange(0, seq_len - 1, device = device)$to(dtype = torch_float())
  position <- position$unsqueeze(2)$unsqueeze(3)

  # div_term: (1, 1, E/2)
  div_term <- torch_exp(
    torch_tensor(seq(0, d_model - 1, by = 2), device = device, dtype = torch_float()) *
      (-log(10000.0) / d_model)
  )
  div_term <- div_term$unsqueeze(1)$unsqueeze(1)

  # Запис у pe: індекси в R 1-based
  pe[ , , seq(1, d_model, by = 2)] <- torch_sin(position * div_term)
  if (d_model >= 2) {
    pe[ , , seq(2, d_model, by = 2)] <- torch_cos(position * div_term)
  }

  pe
}

# --- 3) Модель: Transformer Encoder для next-step prediction ----------------
# Вхід: (B, S) -> (B, S, 1) -> Linear -> (B, S, E) -> permute -> (S, B, E)
# + positional encoding (S, 1, E)
# Encoder -> (S, B, E)
# Беремо останній timestep -> (B, E) -> Linear -> (B, 1)

SimpleTransformerNext <- nn_module(
  "SimpleTransformerNext",
  initialize = function(d_model = 32,
                        nhead = 4,
                        num_layers = 2,
                        dim_ff = 128,
                        dropout = 0.1,
                        seq_len = 40,
                        device = "cpu") {
    self$d_model <- d_model
    self$seq_len <- seq_len

    self$in_proj <- nn_linear(1, d_model)

    enc_layer <- nn_transformer_encoder_layer(
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = dim_ff,
      dropout = dropout
    )
    self$encoder <- nn_transformer_encoder(enc_layer, num_layers = num_layers)

    self$out <- nn_linear(d_model, 1)

    pe <- positional_encoding(seq_len = seq_len, d_model = d_model, device = device)
    self$register_buffer("pe", pe)
  },

  forward = function(x) {
    # x: (B, S)
    x <- x$unsqueeze(3)         # (B, S, 1)
    x <- self$in_proj(x)        # (B, S, E)
    x <- x$permute(c(2, 1, 3))  # (S, B, E)

    x <- x + self$pe            # (S, B, E) + (S, 1, E) -> broadcast по B

    z <- self$encoder(x)        # (S, B, E)

    last <- z[self$seq_len, .., ]  # (B, E)
    self$out(last)                 # (B, 1)
  }
)

# --- 4) Навчання -------------------------------------------------------------
model <- SimpleTransformerNext(
  d_model = 32,
  nhead = 4,
  num_layers = 2,
  dim_ff = 128,
  dropout = 0.1,
  seq_len = seq_len,
  device = device
)
model$to(device = device)

optimizer <- optim_adam(model$parameters, lr = 1e-3)

epochs <- 80
batch_size <- 64

train_losses <- numeric(epochs)
test_losses  <- numeric(epochs)

eval_loss <- function(X, y) {
  model$eval()
  X <- X$to(device = device)
  y <- y$to(device = device)
  pred <- model(X)
  loss <- nnf_mse_loss(pred, y)
  as.numeric(loss$item())
}

for (epoch in 1:epochs) {
  model$train()

  # Перемішування
  idx <- sample.int(n_train)
  Xs <- train_data$X[idx, ]
  ys <- train_data$y[idx, ]

  n_batches <- ceiling(n_train / batch_size)
  epoch_loss_sum <- 0

  for (b in 1:n_batches) {
    start <- (b - 1) * batch_size + 1
    end   <- min(b * batch_size, n_train)

    xb <- Xs[start:end, ]$to(device = device)
    yb <- ys[start:end, ]$to(device = device)

    optimizer$zero_grad()
    pred <- model(xb)
    loss <- nnf_mse_loss(pred, yb)

    loss$backward()
    optimizer$step()

    epoch_loss_sum <- epoch_loss_sum + as.numeric(loss$item())
  }

  train_losses[epoch] <- epoch_loss_sum / n_batches
  test_losses[epoch]  <- eval_loss(test_data$X, test_data$y)

  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d/%d | train_loss=%.6f | test_loss=%.6f\n",
                epoch, epochs, train_losses[epoch], test_losses[epoch]))
  }
}

# --- 5) Графік збіжності (крива втрат) --------------------------------------
plot(1:epochs, train_losses, type = "l",
     xlab = "Epoch", ylab = "MSE Loss",
     main = "Transformer Encoder: крива втрат (train vs test)")
lines(1:epochs, test_losses, lty = 2)
legend("topright", legend = c("Train", "Test"), lty = c(1, 2), bty = "n")

# --- 6) Демонстрація прогнозу наступного кроку ------------------------------
model$eval()

X_demo <- test_data$X[1:10, ]$to(device = device)
y_demo <- test_data$y[1:10, ]$to(device = device)
pred   <- model(X_demo)

cat("\nПерші 10 прогнозів (y_true vs y_pred):\n")
out <- cbind(
  y_true = as.numeric(y_demo$cpu()$squeeze()),
  y_pred = as.numeric(pred$cpu()$squeeze())
)
print(round(out, 4))
