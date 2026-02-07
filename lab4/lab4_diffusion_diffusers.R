############################################################
# Лабораторна робота 4: Дифузійні моделі (diffusers) у R
############################################################

# ----------------------------------------------------------
# 0) R-пакети
# ----------------------------------------------------------
req <- c("reticulate","ggplot2")
to_install <- setdiff(req, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(reticulate)
library(ggplot2)

# ----------------------------------------------------------
# 1) Папка для результатів
# ----------------------------------------------------------
out_dir <- "plots_diffusion"
if (!dir.exists(out_dir)) dir.create(out_dir)

# ----------------------------------------------------------
# 2) Python середовище (venv у папці проєкту)
# ----------------------------------------------------------
venv_path <- "venv_diffusers"
py_bin <- Sys.which("python3")
if (py_bin == "") stop("Не знайдено python3 у PATH. Встанови Python або додай у PATH.")

if (!dir.exists(venv_path)) {
  message("Створюю virtualenv: ", venv_path)
  virtualenv_create(venv_path, python = py_bin)
} else {
  message("virtualenv вже існує: ", venv_path)
}

use_virtualenv(venv_path, required = TRUE)

# ----------------------------------------------------------
# 3) Python пакети
# ----------------------------------------------------------
py_install(c(
  "numpy",
  "Pillow",
  "torch",
  "transformers",
  "accelerate",
  "safetensors",
  "diffusers"
), pip = TRUE)

py_run_string("
import torch, diffusers, transformers
print('[OK] torch:', torch.__version__)
print('[OK] diffusers:', diffusers.__version__)
print('[OK] transformers:', transformers.__version__)
")

# ----------------------------------------------------------
# 4) Налаштування генерації
# ----------------------------------------------------------
# ЛЕГКА модель (швидше завантажується, менше памʼяті)
# Опції:
# 1. SimianLuo/LCM_Dreamshaper_v7 - швидка LCM модель (~2GB)
# 2. segmind/small-sd - мала SD модель (~900MB)
# 3. runwayml/stable-diffusion-v1-5 - стандартна (~4GB)

model_id <- "segmind/small-sd"  # Найменша і найшвидша

# Параметри
seed_base <- 2026L
steps <- 20L  # Менше кроків для швидкості
guidance <- 7.5
img_w <- 512L
img_h <- 512L

# ----------------------------------------------------------
# 5) Python-хелпер з покращеним завантаженням
# ----------------------------------------------------------
py_run_string("
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import os

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def make_generator(device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def load_pipe(model_id):
    device = get_device()
    dtype = torch.float16 if device == 'cuda' else torch.float32

    print(f'[INFO] Завантажую модель: {model_id}')
    print(f'[INFO] Device: {device}, dtype: {dtype}')
    print('[INFO] Це може зайняти 1-5 хвилин залежно від інтернету...')

    # Налаштування для швидшого завантаження
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Вимикаємо hf_transfer

    # Завантажуємо модель
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True  # Економія памʼяті
    )

    if device == 'cuda':
        pipe = pipe.to(device)
    else:
        pipe = pipe.to('cpu')
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()  # Додаткова економія памʼяті
        print('[INFO] CPU mode: увімкнено memory-efficient режим')

    # Scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print('[OK] Pipeline готовий!')
    return pipe

def run_t2i(pipe, prompt, negative_prompt, steps, guidance, seed, width, height):
    device = pipe._execution_device.type
    g = make_generator(device, seed)
    print(f'[GEN] Генерую (seed={seed})...')
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g,
        width=int(width),
        height=int(height)
    )
    print('[OK] ✓')
    return out.images[0]
")

message("\n=== Завантажую ЛЕГКУ модель (segmind/small-sd, ~900MB) ===")
message("Якщо зависне - просто почекайте 2-3 хвилини, файли качаються")

pipe <- py$load_pipe(model_id)

# ----------------------------------------------------------
# 6) Базовий text-to-image
# ----------------------------------------------------------
prompt_base <- "a macro photo of a dew-covered leaf, ultra-detailed, soft morning light"
neg_base <- "blurry, low quality, artifacts, watermark, text"

message("\n=== Генерую baseline зображення ===")
img0 <- py$run_t2i(pipe, prompt_base, neg_base, steps, guidance, seed_base, img_w, img_h)
path0 <- file.path(out_dir, "fig_01_t2i_baseline_leaf.png")
img0$save(path0)
message("✓ Збережено: ", path0)

# ----------------------------------------------------------
# 7) ІНДИВІДУАЛЬНЕ №9: композиція (ракурс/об'єктив)
# ВАЖЛИВО: Той самий seed для всіх - змінюється ТІЛЬКИ геометрія!
# ----------------------------------------------------------
scene <- "a robot barista making coffee in a cozy cafe, warm lighting, detailed"

# ФІКСОВАНИЙ seed для всіх варіантів!
fixed_seed <- seed_base

prompts <- list(
  list(name="fig_02_comp_baseline",
       prompt=scene,
       note="Baseline (без вказівок на композицію)"),

  list(name="fig_03_comp_wide_35mm",
       prompt=paste("wide-angle 35mm lens,", scene),
       note="Wide-angle 35mm (широка перспектива)"),

  list(name="fig_04_comp_tele_85mm",
       prompt=paste("telephoto 85mm lens,", scene),
       note="Telephoto 85mm (стиснута перспектива)"),

  list(name="fig_05_comp_closeup",
       prompt=paste("extreme close-up shot,", scene),
       note="Close-up (великий план)"),

  list(name="fig_06_comp_topdown",
       prompt=paste("top-down view, overhead shot,", scene),
       note="Top-down (вид зверху)"),

  list(name="fig_07_comp_low_angle",
       prompt=paste("low angle shot, looking up,", scene),
       note="Low angle (знизу вгору)")
)

neg <- "blurry, lowres, bad anatomy, artifacts, watermark, text"

message("\n=== Генерую серію композицій (ФІКСОВАНИЙ SEED) ===")
message("Seed для всіх варіантів: ", fixed_seed)
message("Це забезпечить однаковий вміст, але різну геометрію\n")

for (i in seq_along(prompts)) {
  item <- prompts[[i]]
  message(sprintf("[%d/%d] %s", i, length(prompts), item$note))

  # ВСІ використовують ОДИН seed!
  img <- py$run_t2i(pipe, item$prompt, neg, steps, guidance, fixed_seed, img_w, img_h)

  pth <- file.path(out_dir, paste0(item$name, ".png"))
  img$save(pth)
  message("✓ Збережено: ", basename(pth))
}

# ----------------------------------------------------------
# 8) Підсумок
# ----------------------------------------------------------
message("\n=== ГОТОВО ===")
message("Папка: ", out_dir)
all_files <- list.files(out_dir, pattern = "\\.png$")
message("Згенеровано: ", length(all_files), " зображень")
for (f in all_files) message("  - ", f)
