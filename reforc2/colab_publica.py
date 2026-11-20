# colab_publica.py — s'executa al Colab (amb el secret HF_TOKEN): passa el model a bf16, fa la versió MLX 4 bits i puja les fitxes
import json, os, subprocess, sys, urllib.request
from google.colab import userdata
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
api = HfApi(token=userdata.get("HF_TOKEN")); print("com a", api.whoami()["name"])
BF = "xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2"
MLX = "xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2-MLX-4bit"
DS = "xaviviro/sintaxi-catala-destillada-20k-reforc2"
GH = "https://raw.githubusercontent.com/xaviviro/destillacio-sintaxi-catala/main/reforc2/fitxes"
fitxa = lambda n: urllib.request.urlopen(f"{GH}/{n}").read()

# 1. bf16
import torch
from safetensors.torch import load_file, save_file
D = snapshot_download(BF, local_dir="/content/r2", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
if os.path.exists(f"{D}/model.safetensors.index.json"): os.remove(f"{D}/model.safetensors.index.json")
sd = load_file(f"{D}/model.safetensors")
if next(iter(sd.values())).dtype != torch.bfloat16:
    save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, f"{D}/model.safetensors", metadata={"format": "pt"})
del sd
c = json.load(open(f"{D}/config.json")); c["dtype"] = "bfloat16"; c.get("text_config", {})["dtype"] = "bfloat16"
json.dump(c, open(f"{D}/config.json", "w"), indent=2, ensure_ascii=False)
open(f"{D}/README.md", "wb").write(fitxa("model_bf16.md"))
print("bf16:", os.path.getsize(f"{D}/model.safetensors") / 1e9, "GB")
if any(f.rfilename == "model.safetensors.index.json" for f in api.model_info(BF).siblings):
    api.delete_file("model.safetensors.index.json", repo_id=BF, commit_message="Treu un índex de pesos que no correspon")
api.upload_folder(folder_path=D, repo_id=BF, allow_patterns=["model.safetensors", "config.json", "README.md"],
                  commit_message="Pesos en bf16 (1,7 GB) i fitxa del model")

# 2. MLX 4 bits
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlx[cpu]", "mlx-lm"], check=True)
subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", D, "--mlx-path", "/content/r2_mlx", "-q",
                "--q-bits", "4", "--q-group-size", "64"], check=True)
# mateixa configuració que la versió MLX anterior (camps de text al primer nivell + text_config: LM Studio)
cfg = hf_hub_download("xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc-MLX-4bit", "config.json", local_dir="/content/cfg_r1")
os.replace(cfg, "/content/r2_mlx/config.json")
open("/content/r2_mlx/README.md", "wb").write(fitxa("model_mlx4.md"))
print("mlx:", sorted(os.listdir("/content/r2_mlx")), os.path.getsize("/content/r2_mlx/model.safetensors") / 1e6, "MB")
api.create_repo(MLX, exist_ok=True)
api.upload_folder(folder_path="/content/r2_mlx", repo_id=MLX, commit_message="MLX 4 bits del mini 20k + 2 reforços")

# 3. fitxa del dataset
api.upload_file(path_or_fileobj=fitxa("dataset.md"), path_in_repo="README.md", repo_id=DS, repo_type="dataset",
                commit_message="Fitxa: resultats del model entrenat")
for r in (BF, MLX):
    print(r, sorted((f.rfilename, f.size) for f in api.model_info(r, files_metadata=True).siblings))
print("FET")
