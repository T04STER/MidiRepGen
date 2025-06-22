import sys
from src.common.metrics.fid import calculate_fad
from src.common.config_utils import get_config
import os
import json


# using script for that instead of function because ipyrunner breaks the FrechetAudioDistance lib
def main():
    config = get_config("fad")      # using config instead of args because FrechetAudioDistance breaks when using args

    real_path = config["real_wavs"]
    generated_ddpm_path = config["generated_wavs_ddpm"]
    generated_ddim_path = config["generated_wavs_ddim"]
    generated_vae_path = config["generated_wavs_vae"]
    output_fad =  config["output_fad"]

    non_empty_dirs = []
    for path, model_name in zip([generated_ddpm_path, generated_ddim_path, generated_vae_path], ["ddpm", "ddim", "vae"]):
        if os.path.isdir(path) and any(os.scandir(path)):
            non_empty_dirs.append((path, model_name))

    fid_scores =  { }
    for path, model_name in non_empty_dirs:
        print(f"Calculating FAD for {model_name}...")
        fid_score = calculate_fad(real_path, path)
        print(f"FAD score for {model_name}: {fid_score}")
        fid_scores[model_name] = fid_score

    print("FAD scores:", fid_scores)
 
    with open(output_fad, "w") as f:
        json.dump(fid_scores, f, indent=4)

if __name__ == "__main__":
    sys.exit(main())