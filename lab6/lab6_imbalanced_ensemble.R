############################################################
# Лабораторна 6: Ансамблі моделей у Deep Learning / ML (R)
# Індивідуальне завдання №9: незбалансовані дані + F1/ROC-AUC/PR
############################################################

# ----------------------------------------------------------
# 0) R-пакети
# ----------------------------------------------------------
req <- c(
  "caret","caretEnsemble","dplyr","ggplot2","pROC",
  "PRROC","tibble","forcats","reshape2","tidyr"
)

to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(caret)
library(caretEnsemble)
library(dplyr)
library(ggplot2)
library(pROC)
library(PRROC)
library(tibble)
library(forcats)
library(reshape2)
library(tidyr)

# ----------------------------------------------------------
# 1) Вихідна папка
# ----------------------------------------------------------
out_dir <- "plots_ensembles"
if (!dir.exists(out_dir)) dir.create(out_dir)

save_plot <- function(name, p, w=8, h=6, dpi=250) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, p, width=w, height=h, dpi=dpi, bg="white")
  cat("  ✓ Збережено:", path, "\n")
}

# ----------------------------------------------------------
# 2) Дані з дисбалансом (двокласова задача)
#    Використаємо вбудований датасет caret: twoClassSim
# ----------------------------------------------------------
set.seed(42)

n <- 3000
df <- caret::twoClassSim(n = n)

# twoClassSim створює фактор Class із рівнями: "Class1", "Class2"
# Зробимо явні рівні "pos"/"neg" і штучний дисбаланс:
df <- df %>%
  mutate(
    Class = ifelse(Class == "Class1", "pos", "neg"),
    Class = factor(Class, levels = c("pos","neg"))
  )

# Робимо дисбаланс: залишимо всі "pos", а "neg" прорідимо
pos_df <- df %>% filter(Class == "pos")
neg_df <- df %>% filter(Class == "neg") %>% sample_frac(0.15)  # ~15% від neg
df_imb <- bind_rows(pos_df, neg_df) %>% sample_frac(1)

cat("\n=== Розподіл класів (дисбаланс) ===\n")
print(prop.table(table(df_imb$Class)))

# Train/Test split (stratified)
idx <- createDataPartition(df_imb$Class, p = 0.8, list = FALSE)
train <- df_imb[idx,]
test  <- df_imb[-idx,]

cat("\nTrain size:", nrow(train), " | Test size:", nrow(test), "\n")

# ----------------------------------------------------------
# 3) TrainControl: CV + classProbs + summaryFunction для ROC
# ----------------------------------------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = FALSE
)

# ----------------------------------------------------------
# 4) Базові моделі (різні сімейства -> різноманіття помилок)
#    - rf  : bagging-like (Random Forest)
#    - gbm : boosting
#    - glmnet: регуляризована логістична регресія
# ----------------------------------------------------------
set.seed(42)

# Дрібні тюнінги, щоб стабільно бігало
tuneList <- list(
  rf = caretModelSpec(
    method = "rf",
    tuneLength = 3,
    importance = TRUE
  ),
  gbm = caretModelSpec(
    method = "gbm",
    tuneLength = 3,
    verbose = FALSE
  ),
  glmnet = caretModelSpec(
    method = "glmnet",
    tuneLength = 5
  )
)

message("\n=== Треную базові моделі (caretList) ===")
models <- caretList(
  Class ~ .,
  data = train,
  trControl = ctrl,
  metric = "ROC",
  tuneList = tuneList
)

cat("\n=== Базові моделі натреновано ===\n")
print(models)

# ----------------------------------------------------------
# 5) Stacking (meta-learner)
# ----------------------------------------------------------
stack_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

message("\n=== Треную стекінг (caretStack) ===")
stack_model <- caretStack(
  models,
  method = "glm",
  metric = "ROC",
  trControl = stack_ctrl
)

cat("\n=== Stack model ===\n")
print(stack_model)

# ----------------------------------------------------------
# 6) Функції для отримання ймовірностей "pos"
# ----------------------------------------------------------
get_probs_train <- function(model, newdata) {
  p <- predict(model, newdata = newdata, type = "prob")
  as.numeric(p[,"pos"])
}

# Ймовірність "pos" для caretStack
get_probs_stack <- function(stack_model, newdata) {
  # Крок 1: Отримуємо прогнози базових моделей
  base_preds <- lapply(stack_model$models, function(m) {
    predict(m, newdata = newdata, type = "prob")[, "pos"]
  })
  base_preds_df <- as.data.frame(base_preds)
  names(base_preds_df) <- names(stack_model$models)

  # Крок 2: Використовуємо мета-модель (glm)
  # caretStack зберігає фінальну модель у $ens_model$finalModel
  meta_pred <- predict(stack_model$ens_model$finalModel,
                       newdata = base_preds_df,
                       type = "response")
  return(as.numeric(meta_pred))
}

# ----------------------------------------------------------
# 7) Прогнози на test
# ----------------------------------------------------------
# True labels
y_true <- test$Class

# base probs
probs_base <- lapply(models, get_probs_train, newdata = test)
probs_base <- probs_base %>% as_tibble()
names(probs_base) <- names(models)

# stack probs
prob_stack <- get_probs_stack(stack_model, test)

# sanity checks
stopifnot(length(prob_stack) == nrow(test))
stopifnot(all(prob_stack >= 0 & prob_stack <= 1))

# Класи за порогом 0.5 (можна потім оптимізувати по F1)
pred_from_prob <- function(p, thr = 0.5) factor(ifelse(p >= thr, "pos", "neg"), levels=c("pos","neg"))

pred_base  <- lapply(probs_base, pred_from_prob)
pred_stack <- pred_from_prob(prob_stack)

# ----------------------------------------------------------
# 8) Метрики: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
# ----------------------------------------------------------
# Обчислення F1 (pos як позитивний клас)
calc_f1 <- function(y_true, y_pred) {
  cm <- caret::confusionMatrix(y_pred, y_true, positive = "pos")
  prec <- as.numeric(cm$byClass["Precision"])
  rec  <- as.numeric(cm$byClass["Recall"])
  if (is.na(prec) || is.na(rec) || (prec + rec) == 0) return(NA_real_)
  2 * prec * rec / (prec + rec)
}

calc_acc <- function(y_true, y_pred) mean(y_true == y_pred)

calc_roc_auc <- function(y_true, prob_pos) {
  roc_obj <- pROC::roc(response = y_true, predictor = prob_pos, levels = c("neg","pos"), direction = "<", quiet = TRUE)
  as.numeric(pROC::auc(roc_obj))
}

calc_pr_auc <- function(y_true, prob_pos) {
  # PRROC очікує: scores.class0 = позитивний клас
  fg <- prob_pos[y_true == "pos"]
  bg <- prob_pos[y_true == "neg"]
  pr <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = FALSE)
  as.numeric(pr$auc.integral)
}

# Збір метрик у таблицю
rows <- list()

for (m in names(models)) {
  p <- probs_base[[m]]
  yhat <- pred_from_prob(p, 0.5)
  rows[[m]] <- tibble(
    Model = m,
    Accuracy = calc_acc(y_true, yhat),
    F1 = calc_f1(y_true, yhat),
    ROC_AUC = calc_roc_auc(y_true, p),
    PR_AUC  = calc_pr_auc(y_true, p)
  )
}

rows[["stack"]] <- tibble(
  Model = "stack_glm",
  Accuracy = calc_acc(y_true, pred_stack),
  F1 = calc_f1(y_true, pred_stack),
  ROC_AUC = calc_roc_auc(y_true, prob_stack),
  PR_AUC  = calc_pr_auc(y_true, prob_stack)
)

metrics_tbl <- bind_rows(rows) %>%
  arrange(desc(PR_AUC))

cat("\n=== Метрики (чим вище PR_AUC — тим краще для дисбалансу) ===\n")
print(metrics_tbl)

# Збережемо таблицю
write.csv(metrics_tbl, file.path(out_dir, "metrics_table.csv"), row.names = FALSE)
cat("  ✓ Збережено:", file.path(out_dir, "metrics_table.csv"), "\n")

# ----------------------------------------------------------
# 9) ROC-криві (для всіх моделей + stack)
# ----------------------------------------------------------
roc_df_list <- list()

# base
for (m in names(models)) {
  roc_obj <- pROC::roc(response = y_true, predictor = probs_base[[m]],
                       levels = c("neg","pos"), direction = "<", quiet = TRUE)
  coords <- pROC::coords(roc_obj, "all", ret=c("specificity","sensitivity"), transpose = FALSE)
  roc_df_list[[m]] <- tibble(
    Model = m,
    FPR = 1 - coords$specificity,
    TPR = coords$sensitivity
  )
}

# stack
roc_obj_s <- pROC::roc(response = y_true, predictor = prob_stack,
                       levels = c("neg","pos"), direction = "<", quiet = TRUE)
coords_s <- pROC::coords(roc_obj_s, "all", ret=c("specificity","sensitivity"), transpose = FALSE)
roc_df_list[["stack_glm"]] <- tibble(
  Model = "stack_glm",
  FPR = 1 - coords_s$specificity,
  TPR = coords_s$sensitivity
)

roc_df <- bind_rows(roc_df_list)

p_roc <- ggplot(roc_df, aes(FPR, TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = 2) +
  labs(
    title = "ROC-криві: базові моделі vs стекінг",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Модель"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_01_roc_curves", p_roc)

# ----------------------------------------------------------
# 10) PR-криві (важливіше для дисбалансу)
# ----------------------------------------------------------
pr_curve_df <- function(y_true, prob_pos, name) {
  fg <- prob_pos[y_true == "pos"]
  bg <- prob_pos[y_true == "neg"]
  pr <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
  tibble(
    Model = name,
    Recall = pr$curve[,1],
    Precision = pr$curve[,2]
  )
}

pr_df <- bind_rows(
  lapply(names(models), function(m) pr_curve_df(y_true, probs_base[[m]], m)),
  pr_curve_df(y_true, prob_stack, "stack_glm")
)

p_pr <- ggplot(pr_df, aes(Recall, Precision, color = Model)) +
  geom_line(linewidth = 1) +
  labs(
    title = "PR-криві (Precision–Recall): базові моделі vs стекінг",
    x = "Recall",
    y = "Precision",
    color = "Модель"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_02_pr_curves", p_pr)

# ----------------------------------------------------------
# 11) Матриця невідповідностей (Confusion Matrix) для stack (heatmap)
# ----------------------------------------------------------
cm_stack <- caret::confusionMatrix(pred_stack, y_true, positive = "pos")
cat("\n=== Confusion Matrix (stack) ===\n")
print(cm_stack$table)
print(cm_stack$byClass[c("Precision","Recall")])

cm_m <- as.matrix(cm_stack$table)
cm_long <- reshape2::melt(cm_m)
names(cm_long) <- c("Pred","True","Count")

p_cm <- ggplot(cm_long, aes(True, Pred, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), size = 5) +
  labs(
    title = "Матриця невідповідностей (stack_glm)",
    x = "Істинний клас",
    y = "Передбачений клас",
    fill = "К-сть"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_03_confusion_matrix_stack", p_cm)

# ----------------------------------------------------------
# 12) Порівняльний барплот метрик (PR_AUC, ROC_AUC, F1)
# ----------------------------------------------------------
metrics_long <- metrics_tbl %>%
  pivot_longer(cols = c("PR_AUC","ROC_AUC","F1"), names_to = "Metric", values_to = "Value")

p_bar <- ggplot(metrics_long, aes(fct_reorder(Model, Value), Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  labs(
    title = "Порівняння метрик моделей (особливо важливий PR_AUC для дисбалансу)",
    x = "Модель",
    y = "Значення",
    fill = "Метрика"
  ) +
  theme_minimal(base_size = 13)

save_plot("fig_04_metrics_barplot", p_bar, w=10, h=6)

cat("\n=== ГОТОВО ===\n")
cat("Папка з результатами:", out_dir, "\n")
print(list.files(out_dir))
