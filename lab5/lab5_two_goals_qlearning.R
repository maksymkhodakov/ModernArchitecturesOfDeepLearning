############################################################
# Лабораторна робота 5 (Індивідуальне №9)
# Дві цілі з різними винагородами: Q-learning у GridWorld
############################################################

# ----------------------------------------------------------
# 0) Пакети
# ----------------------------------------------------------
req <- c("ggplot2", "dplyr", "tidyr", "scales")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ----------------------------------------------------------
# 1) Папки для результатів
# ----------------------------------------------------------
dir.create("plots", showWarnings = FALSE)
dir.create("tables", showWarnings = FALSE)

# ----------------------------------------------------------
# 2) Опис середовища GridWorld з ДВОМА цілями
# ----------------------------------------------------------
# Сітка 7x7, старт у центрі.
# goal2 (ближче, дешевше) = +0.5
# goal1 (далі, дорожче)   = +1.0
#
# Кроковий штраф (step-cost): -0.02
# Це важливо, щоб стимулювати коротші траєкторії
# і зробити компроміс відчутним при різних gamma.
make_env <- function(n = 7) {
  env <- list()
  env$n <- n
  env$start <- c(4, 4)      # (row, col)
  env$goal2 <- c(4, 6)      # ближча
  env$goal1 <- c(7, 7)      # дальша

  env$r_goal2 <- +0.5
  env$r_goal1 <- +1.0
  env$step_cost <- -0.02

  env$max_steps <- 200
  env
}

# Перетворення (row,col) <-> state_id
# state_id: 1..(n*n)
rc_to_s <- function(r, c, n) (r - 1) * n + c
s_to_rc <- function(s, n) c(((s - 1) %/% n) + 1, ((s - 1) %% n) + 1)

# Дії: 1=Up, 2=Right, 3=Down, 4=Left
ACTIONS <- c("U","R","D","L")

step_env <- function(env, state, action) {
  n <- env$n
  rc <- s_to_rc(state, n)
  r <- rc[1]; c <- rc[2]

  r2 <- r; c2 <- c
  if (action == 1) r2 <- r - 1
  if (action == 2) c2 <- c + 1
  if (action == 3) r2 <- r + 1
  if (action == 4) c2 <- c - 1

  # межі
  r2 <- max(1, min(n, r2))
  c2 <- max(1, min(n, c2))

  next_state <- rc_to_s(r2, c2, n)

  # винагорода
  reward <- env$step_cost
  done <- FALSE
  goal_hit <- NA_character_

  if (r2 == env$goal2[1] && c2 == env$goal2[2]) {
    reward <- reward + env$r_goal2
    done <- TRUE
    goal_hit <- "goal2(+0.5)"
  }
  if (r2 == env$goal1[1] && c2 == env$goal1[2]) {
    reward <- reward + env$r_goal1
    done <- TRUE
    goal_hit <- "goal1(+1.0)"
  }

  list(next_state = next_state, reward = reward, done = done, goal_hit = goal_hit)
}

# ----------------------------------------------------------
# 3) Q-learning (табличний)
# ----------------------------------------------------------
epsilon_greedy <- function(Q_row, eps) {
  if (runif(1) < eps) {
    sample.int(length(Q_row), 1)
  } else {
    # випадково серед максимальних
    mx <- max(Q_row)
    best <- which(Q_row == mx)
    sample(best, 1)
  }
}

train_qlearning <- function(env,
                            episodes = 4000,
                            alpha = 0.10,
                            gamma = 0.90,
                            eps_start = 0.80,
                            eps_end = 0.05,
                            eps_decay = 0.999,
                            seed = 42) {
  set.seed(seed)

  nS <- env$n * env$n
  nA <- 4
  Q <- matrix(0, nrow = nS, ncol = nA)

  eps <- eps_start

  # лог кривих
  log_df <- data.frame(
    episode = integer(episodes),
    G = numeric(episodes),          # return (дисконтований)
    steps = integer(episodes),
    reached = character(episodes),
    stringsAsFactors = FALSE
  )

  start_state <- rc_to_s(env$start[1], env$start[2], env$n)

  for (ep in 1:episodes) {
    s <- start_state
    done <- FALSE
    t <- 0
    G <- 0
    reached <- "none"

    while (!done && t < env$max_steps) {
      t <- t + 1
      a <- epsilon_greedy(Q[s, ], eps)

      tr <- step_env(env, s, a)
      s2 <- tr$next_state
      r <- tr$reward

      # Q-learning update:
      # Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_a' Q(s',a') - Q(s,a))
      Q[s, a] <- Q[s, a] + alpha * (r + gamma * max(Q[s2, ]) - Q[s, a])

      # return
      G <- G + (gamma^(t-1)) * r

      s <- s2
      done <- tr$done
      if (done) reached <- tr$goal_hit
    }

    log_df$episode[ep] <- ep
    log_df$G[ep] <- G
    log_df$steps[ep] <- t
    log_df$reached[ep] <- reached

    # epsilon decay
    eps <- max(eps_end, eps * eps_decay)
  }

  list(Q = Q, log = log_df)
}

# ----------------------------------------------------------
# 4) Оцінювання політики: greedy rollout
# ----------------------------------------------------------
run_greedy_episode <- function(env, Q, gamma = 0.9) {
  n <- env$n
  start_state <- rc_to_s(env$start[1], env$start[2], n)

  s <- start_state
  done <- FALSE
  t <- 0
  G <- 0
  reached <- "none"
  path <- data.frame(step = 0, state = s, r = env$start[1], c = env$start[2])

  while (!done && t < env$max_steps) {
    t <- t + 1
    # greedy action (tie-break random)
    mx <- max(Q[s, ])
    best <- which(Q[s, ] == mx)
    a <- sample(best, 1)

    tr <- step_env(env, s, a)
    s2 <- tr$next_state
    rc2 <- s_to_rc(s2, n)

    G <- G + (gamma^(t-1)) * tr$reward

    path <- rbind(path, data.frame(step = t, state = s2, r = rc2[1], c = rc2[2]))

    s <- s2
    done <- tr$done
    if (done) reached <- tr$goal_hit
  }

  list(reached = reached, steps = t, G = G, path = path)
}

evaluate_goal_choice <- function(env, Q, gamma, n_eval = 400, seed = 999) {
  set.seed(seed)
  res <- replicate(n_eval, run_greedy_episode(env, Q, gamma = gamma), simplify = FALSE)

  reached <- sapply(res, `[[`, "reached")
  steps <- sapply(res, `[[`, "steps")
  G <- sapply(res, `[[`, "G")

  tibble(
    gamma = gamma,
    n_eval = n_eval,
    p_goal1 = mean(reached == "goal1(+1.0)"),
    p_goal2 = mean(reached == "goal2(+0.5)"),
    p_none  = mean(reached == "none"),
    avg_steps = mean(steps),
    avg_return = mean(G)
  )
}

# ----------------------------------------------------------
# 5) Візуалізація: policy + value heatmap (для кожного gamma)
# ----------------------------------------------------------
# value = max_a Q(s,a)
# policy = argmax_a Q(s,a)
policy_df <- function(env, Q) {
  n <- env$n
  nS <- n*n

  df <- lapply(1:nS, function(s) {
    rc <- s_to_rc(s, n)
    v <- max(Q[s, ])
    a <- which(Q[s, ] == max(Q[s, ]))
    a <- sample(a, 1)  # tie-break
    data.frame(
      s = s,
      r = rc[1],
      c = rc[2],
      V = v,
      a = a
    )
  }) |> bind_rows()

  # для красивої карти: y зверху вниз
  df$y_plot <- n - df$r + 1
  df$x_plot <- df$c

  df
}

arrow_segment_df <- function(df_pol) {
  # напрямки в координатах ggplot: x вправо, y вгору
  # U: (0,+0.35), R:(+0.35,0), D:(0,-0.35), L:(-0.35,0)
  dx <- c(0, +0.35, 0, -0.35)
  dy <- c(+0.35, 0, -0.35, 0)

  df_pol |>
    mutate(
      x = x_plot, y = y_plot,
      xend = x_plot + dx[a],
      yend = y_plot + dy[a]
    )
}

plot_policy_value <- function(env, Q, gamma_label) {
  df <- policy_df(env, Q)
  arr <- arrow_segment_df(df)

  # позначимо старт і цілі
  n <- env$n
  start <- data.frame(
    x_plot = env$start[2], y_plot = n - env$start[1] + 1,
    label = "START"
  )
  g1 <- data.frame(
    x_plot = env$goal1[2], y_plot = n - env$goal1[1] + 1,
    label = "goal1(+1.0)"
  )
  g2 <- data.frame(
    x_plot = env$goal2[2], y_plot = n - env$goal2[1] + 1,
    label = "goal2(+0.5)"
  )

  ggplot(df, aes(x_plot, y_plot)) +
    geom_tile(aes(fill = V), color = "white", linewidth = 0.4) +
    geom_segment(
      data = arr,
      aes(x = x, y = y, xend = xend, yend = yend),
      arrow = arrow(length = unit(0.12, "inches")),
      linewidth = 0.6,
      alpha = 0.85
    ) +
    geom_point(data = start, aes(x_plot, y_plot), size = 4) +
    geom_text(data = start, aes(x_plot, y_plot, label = label), vjust = -1.1, fontface = 2) +
    geom_point(data = g1, aes(x_plot, y_plot), size = 4) +
    geom_text(data = g1, aes(x_plot, y_plot, label = label), vjust = -1.1) +
    geom_point(data = g2, aes(x_plot, y_plot), size = 4) +
    geom_text(data = g2, aes(x_plot, y_plot, label = label), vjust = -1.1) +
    coord_fixed() +
    scale_x_continuous(breaks = 1:n, name = "X (колонка)") +
    scale_y_continuous(breaks = 1:n, name = "Y (рядок)") +
    labs(
      title = paste0("Політика (стрілки) + карта значень V(s) після навчання | ", gamma_label),
      fill = "V(s)=max Q"
    ) +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
}

# ----------------------------------------------------------
# 6) Експеримент: різні gamma
# ----------------------------------------------------------
env <- make_env(n = 7)

# Конфігурація навчання
CFG <- list(
  episodes   = 5000,
  alpha      = 0.10,
  eps_start  = 0.80,
  eps_end    = 0.05,
  eps_decay  = 0.999,
  seed_train = 2026,
  n_eval     = 500,
  seed_eval  = 777
)

gammas <- c(0.60, 0.80, 0.95)

all_runs <- list()
all_eval <- list()

for (g in gammas) {
  cat("\n=== TRAIN gamma =", g, "===\n")

  run <- train_qlearning(
    env = env,
    episodes = CFG$episodes,
    alpha = CFG$alpha,
    gamma = g,
    eps_start = CFG$eps_start,
    eps_end = CFG$eps_end,
    eps_decay = CFG$eps_decay,
    seed = CFG$seed_train + as.integer(g*100) # щоб різні gamma мали стабільно різні seeds
  )

  ev <- evaluate_goal_choice(
    env = env,
    Q = run$Q,
    gamma = g,
    n_eval = CFG$n_eval,
    seed = CFG$seed_eval + as.integer(g*100)
  )

  run$gamma <- g
  all_runs[[as.character(g)]] <- run
  all_eval[[as.character(g)]] <- ev
}

eval_tbl <- bind_rows(all_eval)

# Збережемо таблицю вибору цілей
write.csv(eval_tbl, file = "tables/goal_choice_table.csv", row.names = FALSE)
cat("\n✓ Таблиця:", "tables/goal_choice_table.csv\n")

# ----------------------------------------------------------
# 7) Графік 1: Частка вибору цілей vs gamma
# ----------------------------------------------------------
eval_long <- eval_tbl |>
  select(gamma, p_goal1, p_goal2, p_none) |>
  pivot_longer(cols = starts_with("p_"),
               names_to = "metric",
               values_to = "value") |>
  mutate(metric = recode(metric,
                         p_goal1 = "goal1(+1.0) (далі)",
                         p_goal2 = "goal2(+0.5) (ближче)",
                         p_none  = "не досягнуто"))


p_choice <- ggplot(eval_long, aes(x = gamma, y = value, group = metric, color = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_x_continuous(breaks = gammas) +
  labs(
    title = "Частка епізодів з вибором кожної цілі після навчання (greedy-policy)",
    subtitle = paste0("Оцінка: n_eval=", CFG$n_eval, " | alpha=", CFG$alpha,
                      " | eps: ", CFG$eps_start, "→", CFG$eps_end, " (decay ", CFG$eps_decay, ")",
                      " | episodes=", CFG$episodes),
    x = expression(gamma~"(дисконт)"),
    y = "Частка епізодів",
    color = "Що обрано"
  ) +
  theme_minimal(base_size = 12)

ggsave("plots/goal_choice_vs_gamma.png", p_choice, width = 11, height = 6, dpi = 250, bg = "white")
cat("✓ Графік:", "plots/goal_choice_vs_gamma.png\n")

# ----------------------------------------------------------
# 8) Графік 2: Криві навчання (return) для різних gamma
# ----------------------------------------------------------
# Для читабельності: згладження ковзним середнім
roll_mean <- function(x, k = 200) {
  # просте згладження (вікно k)
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in 1:n) {
    lo <- max(1, i - k + 1)
    out[i] <- mean(x[lo:i])
  }
  out
}

log_all <- bind_rows(lapply(all_runs, function(run) {
  df <- run$log
  df$gamma <- run$gamma
  df$G_smooth <- roll_mean(df$G, k = 200)
  df
}))

p_learn <- ggplot(log_all, aes(x = episode, y = G_smooth, color = factor(gamma))) +
  geom_line(linewidth = 1) +
  labs(
    title = "Криві навчання Q-learning (згладжений return) для різних γ",
    subtitle = paste0("Згладження: ковзне середнє (k=200) | episodes=", CFG$episodes,
                      " | alpha=", CFG$alpha, " | eps decay=", CFG$eps_decay),
    x = "Епізод",
    y = "Середній дисконтований return (згладжений)",
    color = expression(gamma)
  ) +
  theme_minimal(base_size = 12)

ggsave("plots/learning_curves_by_gamma.png", p_learn, width = 11, height = 6, dpi = 250, bg = "white")
cat("✓ Графік:", "plots/learning_curves_by_gamma.png\n")

# ----------------------------------------------------------
# 9) Артефакт: two_goals_policy.png (політика+цінності для кожного gamma)
# ----------------------------------------------------------
# Зробимо 3 панелі (по одній на gamma) і збережемо в один PNG.
plots_pol <- lapply(gammas, function(g) {
  Q <- all_runs[[as.character(g)]]$Q
  plot_policy_value(env, Q, gamma_label = paste0("gamma=", g))
})

# Без додаткових пакетів (patchwork/cowplot) —
# зберемо через базовий grDevices + grid.
png("plots/two_goals_policy.png", width = 2000, height = 700, res = 200)
par(mfrow = c(1, length(gammas)), mar = c(4, 4, 4, 6))
for (i in seq_along(gammas)) {
  print(plots_pol[[i]])
}
dev.off()
cat("✓ Артефакт:", "plots/two_goals_policy.png\n")

# ----------------------------------------------------------
# 10) Підсумковий друк у консоль
# ----------------------------------------------------------
cat("\n=== ПІДСУМОК (таблиця вибору цілей) ===\n")
print(eval_tbl)

cat("\nГотово.\nФайли:\n")
print(list.files("plots"))
print(list.files("tables"))
