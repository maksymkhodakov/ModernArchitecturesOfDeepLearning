############################################################
# Лабораторна 3: GNN у R через Python (reticulate) на macOS
# PyTorch Geometric (PyG): Karate Club, GCN vs GAT, Attention
############################################################

# ----------------------------------------------------------
# 0) Пакети R
# ----------------------------------------------------------
req <- c("reticulate","igraph","ggraph","ggplot2","dplyr")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(reticulate)
library(igraph)
library(ggraph)
library(ggplot2)
library(dplyr)

# ----------------------------------------------------------
# 0.1) Попередження про конфлікт R-пакета torch
# ----------------------------------------------------------
if ("torch" %in% rownames(installed.packages())) {
  message("УВАГА: у тебе встановлений R-пакет 'torch'. Він може конфліктувати з Python torch через libtorch dylib.")
  message("Не роби library(torch). Якщо робив — перезапусти R (Session -> Restart R) і запусти цей скрипт знову.")
}

# Папка для результатів
out_dir <- "plots_gnn"
if (!dir.exists(out_dir)) dir.create(out_dir)

save_plot <- function(name, p, w=9, h=6, dpi=250) {
  # Зберігаємо PNG (високої якості, працює завжди)
  png_path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(png_path, p, width=w, height=h, dpi=dpi, bg="white")
  cat("  ✓ Збережено:", name, ".png\n")
}

# ----------------------------------------------------------
# 1) Conda env: створити якщо нема + активувати
# ----------------------------------------------------------
env <- "r-gnn"

message("conda_binary(): ", conda_binary())

envs <- conda_list()
if (!(env %in% envs$name)) {
  message("Створюю conda env '", env, "' (python=3.10)...")
  conda_create(envname = env, packages = "python=3.10")
} else {
  message("Conda env '", env, "' вже існує — ок.")
}

use_condaenv(env, required = TRUE)
cfg <- py_config()
print(cfg)

# ----------------------------------------------------------
# 2) КРИТИЧНО для macOS: ізоляція dylib-пошуку
# ----------------------------------------------------------
conda_prefix <- dirname(dirname(cfg$python))
conda_lib <- file.path(conda_prefix, "lib")

message("Conda prefix: ", conda_prefix)
message("Conda lib:    ", conda_lib)

Sys.setenv(DYLD_FALLBACK_LIBRARY_PATH = conda_lib)
Sys.setenv(DYLD_LIBRARY_PATH = "")

# ----------------------------------------------------------
# 2.5) Встановлюємо PyTorch 2.4.0 (має кращу підтримку PyG)
# ----------------------------------------------------------
message("\n=== КРОК 1: Встановлення PyTorch 2.4.0 ===")
system2(cfg$python, c("-m", "pip", "install",
                      "torch==2.4.0", "torchvision", "torchaudio"),
        stdout = TRUE, stderr = TRUE)
message("PyTorch 2.4.0 встановлено!\n")

# ----------------------------------------------------------
# 3) Python-установка: numpy/networkx/pyg
# ----------------------------------------------------------
message("\n=== КРОК 2: Встановлення PyG та залежностей ===")

py_run_string("
import sys, subprocess

def pip_install(*args):
    subprocess.check_call([sys.executable,'-m','pip','install',*args])

# База
try:
    import numpy
    print('[OK] numpy')
except:
    pip_install('numpy')
    print('[OK] numpy installed')

try:
    import networkx
    print('[OK] networkx')
except:
    pip_install('networkx')
    print('[OK] networkx installed')

# Перевіряємо torch
import torch
print(f'[OK] torch версія: {torch.__version__}')

# PyG: З версії 2.3+ можна використовувати БЕЗ torch-scatter/sparse!
# Ці пакети потрібні лише для специфічних операцій
try:
    import torch_geometric
    print(f'[OK] torch_geometric: {torch_geometric.__version__}')
except:
    print('[INSTALL] torch_geometric (базова версія без додаткових залежностей)...')
    pip_install('torch-geometric')
    import torch_geometric
    print(f'[OK] torch_geometric: {torch_geometric.__version__}')

print('[INFO] PyG встановлено в базовій конфігурації (достатньо для GCN/GAT)')
")

message("\n=== КРОК 3: Підготовка даних та навчання ===\n")

# ----------------------------------------------------------
# 4) Karate Club -> PyG Data
# ----------------------------------------------------------
py_run_string("
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

G = nx.karate_club_graph()
n = G.number_of_nodes()

deg = np.array([d for _, d in G.degree()], dtype=np.float32).reshape(-1,1)
clu = np.array([nx.clustering(G, i) for i in range(n)], dtype=np.float32).reshape(-1,1)

x = np.hstack([deg, clu])
x = (x - x.mean(0)) / (x.std(0) + 1e-6)

y = np.array([0 if G.nodes[i]['club']=='Mr. Hi' else 1 for i in range(n)], dtype=np.int64)

edges = np.array(list(G.edges()), dtype=np.int64)
edge_index = edges.T
rev = edge_index[[1,0], :]
edge_index = np.concatenate([edge_index, rev], axis=1)

data = Data(
    x=torch.tensor(x, dtype=torch.float32),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    y=torch.tensor(y, dtype=torch.long)
)

perm = torch.randperm(n)
n_train = int(0.6*n); n_val = int(0.2*n)

train_idx = perm[:n_train]
val_idx   = perm[n_train:n_train+n_val]
test_idx  = perm[n_train+n_val:]

data.train_mask = torch.zeros(n, dtype=torch.bool); data.train_mask[train_idx] = True
data.val_mask   = torch.zeros(n, dtype=torch.bool); data.val_mask[val_idx]     = True
data.test_mask  = torch.zeros(n, dtype=torch.bool); data.test_mask[test_idx]   = True

print('[INFO] data:', data)
")

message("Shapes: ",
        paste(py_eval("tuple(data.x.shape)"), collapse=" "),
        " | ",
        paste(py_eval("tuple(data.edge_index.shape)"), collapse=" "),
        " | ",
        paste(py_eval("tuple(data.y.shape)"), collapse=" "))

# ----------------------------------------------------------
# 5) GCN: тренування + ембеддинги
# ----------------------------------------------------------
py_run_string("
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out

model_gcn = GCN(in_dim=data.x.size(1), hidden=16,
                out_dim=int(data.y.max().item()+1), dropout=0.3)

opt = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=5e-4)

def acc_on(mask):
    model_gcn.eval()
    with torch.no_grad():
        logits = model_gcn(data.x, data.edge_index)
        pred = logits.argmax(-1)
        return (pred[mask] == data.y[mask]).float().mean().item()

best_val = -1.0
best_state = None

for epoch in range(1, 401):
    model_gcn.train()
    logits = model_gcn(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    opt.zero_grad(); loss.backward(); opt.step()

    val_acc = acc_on(data.val_mask)
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.cpu().clone() for k, v in model_gcn.state_dict().items()}

    if epoch % 50 == 0:
        print(f'[GCN] epoch {epoch:3d} | train acc {acc_on(data.train_mask):.2f} | val acc {val_acc:.2f}')

if best_state is not None:
    model_gcn.load_state_dict(best_state)

test_acc = acc_on(data.test_mask)
print('[GCN] test acc:', round(test_acc, 3))

model_gcn.eval()
with torch.no_grad():
    logits = model_gcn(data.x, data.edge_index)
    pred_gcn = logits.argmax(-1).cpu().numpy()
    y_all = data.y.cpu().numpy()

    h1 = model_gcn.conv1(data.x, data.edge_index)
    h1 = F.relu(h1).cpu().numpy()

h1_gcn = h1
")

# ----------------------------------------------------------
# 6) GAT: тренування + увага (індивідуальне №9)
# ----------------------------------------------------------
py_run_string("
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=4, dropout=0.4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden*heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gat2(h, edge_index)
        return out

gat = GAT(in_dim=data.x.size(1), hidden=8,
          out_dim=int(data.y.max().item()+1), heads=4, dropout=0.4)

opt2 = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=5e-4)

def acc_on(mask):
    gat.eval()
    with torch.no_grad():
        logits = gat(data.x, data.edge_index)
        pred = logits.argmax(-1)
        return (pred[mask] == data.y[mask]).float().mean().item()

best_val = -1.0
best_state = None

for epoch in range(1, 401):
    gat.train()
    logits = gat(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    opt2.zero_grad(); loss.backward(); opt2.step()

    val_acc = acc_on(data.val_mask)
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.cpu().clone() for k, v in gat.state_dict().items()}

    if epoch % 50 == 0:
        print(f'[GAT] epoch {epoch:3d} | val acc {val_acc:.2f}')

if best_state is not None:
    gat.load_state_dict(best_state)

test_acc = acc_on(data.test_mask)
print('[GAT] test acc:', round(test_acc, 3))

gat.eval()
with torch.no_grad():
    logits = gat(data.x, data.edge_index)
    pred_gat = logits.argmax(-1).cpu().numpy()

# Увага з першого шару: return_attention_weights=True
gat.eval()
with torch.no_grad():
    h, (ei_att, alpha) = gat.gat1(data.x, data.edge_index, return_attention_weights=True)

alpha_mean = alpha.mean(dim=1).cpu().numpy()
ei_att_np = ei_att.cpu().numpy()
")

# ----------------------------------------------------------
# 7) Повертаємо дані в R
# ----------------------------------------------------------
y_all    <- unlist(py_eval("y_all.tolist()"))
pred_gcn <- unlist(py_eval("pred_gcn.tolist()"))
pred_gat <- unlist(py_eval("pred_gat.tolist()"))
edge_idx <- py_eval("data.edge_index.cpu().numpy().tolist()")
h1_gcn   <- py_eval("h1_gcn.tolist()")

ei_att     <- py_eval("ei_att_np.tolist()")
alpha_mean <- unlist(py_eval("alpha_mean.tolist()"))

n_nodes <- length(y_all)

edges_df <- data.frame(
  from = unlist(edge_idx[[1]]) + 1L,
  to   = unlist(edge_idx[[2]]) + 1L
)

edges_df$from <- pmin(pmax(edges_df$from, 1L), n_nodes)
edges_df$to   <- pmin(pmax(edges_df$to,   1L), n_nodes)

g <- graph_from_data_frame(edges_df, directed = FALSE, vertices = data.frame(name = 1:n_nodes))

set.seed(42)
# Додаємо атрибути до вершин графа ПЕРЕД створенням layout
V(g)$y_true <- factor(y_all)
V(g)$y_pred_gcn <- factor(pred_gcn)
V(g)$y_pred_gat <- factor(pred_gat)

# Тепер створюємо layout - він автоматично включить атрибути вершин
lay <- ggraph::create_layout(g, layout = "fr")

# Перевіряємо що все створено
cat("Колонки layout:", paste(names(lay), collapse=", "), "\n")
cat("Перші рядки:\n")
print(head(lay[, c("name", "y_true", "y_pred_gcn", "y_pred_gat")]))

# ----------------------------------------------------------
# 8) Графіки: істинні / GCN / GAT
# ----------------------------------------------------------
p_true <- ggraph(lay) +
  geom_edge_link(alpha = 0.18) +
  geom_node_point(aes(color = y_true), size = 4) +
  ggtitle("Karate Club — істинні мітки") +
  theme_graph()

p_gcn <- ggraph(lay) +
  geom_edge_link(alpha = 0.18) +
  geom_node_point(aes(color = y_pred_gcn), size = 4) +
  ggtitle("Karate Club — передбачені мітки (GCN)") +
  theme_graph()

p_gat <- ggraph(lay) +
  geom_edge_link(alpha = 0.18) +
  geom_node_point(aes(color = y_pred_gat), size = 4) +
  ggtitle("Karate Club — передбачені мітки (GAT)") +
  theme_graph()

print(p_true); print(p_gcn); print(p_gat)

save_plot("fig_01_true_labels", p_true)
save_plot("fig_02_pred_gcn", p_gcn)
save_plot("fig_03_pred_gat", p_gat)

# ----------------------------------------------------------
# 9) Метрики: Accuracy + Macro-F1
# ----------------------------------------------------------
# Створюємо df_nodes для таблиць (беремо з layout)
df_nodes <- as.data.frame(lay)

tab_gcn <- table(True = df_nodes$y_true, Pred = df_nodes$y_pred_gcn)
tab_gat <- table(True = df_nodes$y_true, Pred = df_nodes$y_pred_gat)

acc <- function(tab) sum(diag(tab)) / sum(tab)

macro_f1 <- function(tab) {
  classes <- intersect(rownames(tab), colnames(tab))
  f1s <- sapply(classes, function(cl){
    TP <- tab[cl, cl]
    FP <- sum(tab[, cl]) - TP
    FN <- sum(tab[cl, ]) - TP
    denom <- (2*TP + FP + FN)
    if (denom == 0) return(0)
    (2*TP) / denom
  })
  mean(f1s)
}

cat("\n=== Матриця помилок GCN ===\n"); print(tab_gcn)
cat("\n=== Матриця помилок GAT ===\n"); print(tab_gat)
cat(sprintf("\nAccuracy: GCN=%.3f | GAT=%.3f\n", acc(tab_gcn), acc(tab_gat)))
cat(sprintf("Macro-F1:  GCN=%.3f | GAT=%.3f\n", macro_f1(tab_gcn), macro_f1(tab_gat)))

# ----------------------------------------------------------
# 10) Індивідуальне №9: увага на ребрах (товщина/прозорість ∝ attention)
# ----------------------------------------------------------
att_df <- data.frame(
  from = unlist(ei_att[[1]]) + 1L,
  to   = unlist(ei_att[[2]]) + 1L,
  alpha = alpha_mean
)

att_df$alpha01 <- (att_df$alpha - min(att_df$alpha)) / (max(att_df$alpha) - min(att_df$alpha) + 1e-9)

att_df2 <- att_df %>%
  mutate(a = pmin(from,to), b = pmax(from,to)) %>%
  group_by(a,b) %>%
  summarise(alpha01 = mean(alpha01), .groups="drop") %>%
  rename(from=a, to=b)

g_att <- g
E(g_att)$alpha01 <- 0

for (i in seq_len(nrow(att_df2))) {
  e_id <- get_edge_ids(g_att, vp = c(att_df2$from[i], att_df2$to[i]), directed = FALSE)
  if (e_id != 0) E(g_att)$alpha01[e_id] <- att_df2$alpha01[i]
}

lay_att <- ggraph::create_layout(g_att, layout = "fr")

p_att <- ggraph(lay_att) +
  geom_edge_link(aes(alpha = alpha01, width = alpha01), show.legend = TRUE) +
  geom_node_point(aes(color = factor(y_all)), size = 4) +
  ggtitle("GAT: увага на ребрах (товщина/прозорість ∝ увазі)") +
  theme_graph()

print(p_att)
save_plot("fig_04_gat_attention_edges", p_att)

# ----------------------------------------------------------
# 11) PCA ембеддингів (GCN conv1) — для візуалізації у 2D
# ----------------------------------------------------------
h1_mat <- do.call(rbind, h1_gcn)
pc <- prcomp(h1_mat, scale. = TRUE)
emb <- data.frame(PC1 = pc$x[,1], PC2 = pc$x[,2], y = factor(y_all))

p_emb <- ggplot(emb, aes(PC1, PC2, color = y)) +
  geom_point(size=3, alpha=0.85) +
  labs(title="GCN embeddings (conv1) — PCA у 2D", color="Клас") +
  theme_minimal(base_size=13)

print(p_emb)
save_plot("fig_05_embeddings_pca", p_emb, w=8, h=6)

cat("\nГотово. Файли у папці: ", out_dir, "\n", sep="")
print(list.files(out_dir))
