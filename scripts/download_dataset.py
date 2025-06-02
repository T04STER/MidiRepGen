import gdown
from src.common.config_utils import get_config


def main():
    config = get_config()
    dataset_config = config["dataset"]
    url = dataset_config["gdrive_url"]
    output_path = dataset_config["out_zip"]
    out_unpacked = dataset_config["out_unpacked"]
    print(f"Downloading dataset from {url} to {output_path}")
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    print(f"Unpacking dataset to {out_unpacked}")
    gdown.extractall(output_path, out_unpacked)
    print("Dataset downloaded and unpacked successfully.")


if __name__ == "__main__":
    main()
    