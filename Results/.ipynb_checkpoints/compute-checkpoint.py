import os, json, sys, torch
from metrics import compute_metrics, compute_metrics_hf
from metrics import save_results

# ===================================================
# ===================================================

dir = "results1k/"
results = []

files = os.listdir (dir)
for fi in files:
    file = dir + f"{fi}"
    if (file.endswith (".json") and "llava_" in fi):
        result = compute_metrics (file)
        result_hf = compute_metrics_hf (file)
        result.update (result_hf)
        result["method"] = fi
        results.append (result)
save_results (results, "results1k.json")

# ===================================================
# ===================================================

dir = "results1k/"
results = []

files = os.listdir (dir)
for fi in files:
    file = dir + f"{fi}"
    if (file.endswith (".json")):
        if (file.endswith (".json") and "llavachef_" in fi):
            result = compute_metrics (file)
            result_hf = compute_metrics_hf (file)
            result.update (result_hf)
            result["method"] = fi
            results.append (result)
save_results (results, "results1k_llavachefv1.json")

# ===================================================
# ===================================================
results = []
save_dir = "results5k/"
files = os.listdir (save_dir)

N = -1
for fi in files:
    file = save_dir + f"{fi}"
    if (file.endswith (".json")):
        result = compute_metrics (file, N = N)
        result_hf = compute_metrics_hf (file, N = N)
        result.update (result_hf)
        result["method"] = fi
        results.append (result)

 
save_results (results, "results5k.json")


# ===================================================
# ===================================================
results = []
save_dir = "results/"
files = os.listdir (save_dir)

N = -1
for fi in files:
    file = save_dir + f"{fi}"
    if (file.endswith (".json")):
        result = compute_metrics (file, N = N)
        try:
            result_hf = compute_metrics_hf (file, N = N)
            result.update (result_hf)
        except:
            pass
        result["method"] = fi
        results.append (result)

 
save_results (results, "results.json")

# ===================================================
# ===================================================