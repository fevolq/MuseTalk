import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
import gdown

from configs import config


def download_from_hf(repo_id, local_dir, *, force_download=False, revision=None):
    if os.path.exists(local_dir) and not force_download:
        return
    print(f'Download {repo_id}')
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            max_workers=8,
            force_download=True,
            revision=revision,
        )
    except Exception as e:
        print(f'An error occurred while downloading {repo_id}: {str(e)}')
        shutil.rmtree(local_dir)


def download_file(url, file_path, name: str):
    if os.path.exists(file_path):
        return
    print(f'Download {name}')
    os.makedirs(str(Path(file_path).parent), exist_ok=True)
    gdown.download(url, file_path, quiet=False)


def download_models():
    print('Download checkpoint...')
    download_from_hf('TMElyralab/MuseTalk', os.path.join(config.MODEL_DIR, 'MuseTalk'), revision='5e6f29e')
    download_from_hf('stabilityai/sd-vae-ft-mse', os.path.join(config.MODEL_DIR, 'sd-vae-ft-mse'))  # weight
    download_from_hf('yzd-v/DWPose', os.path.join(config.MODEL_DIR, 'dwpose'))  # dwpose

    download_file(
        'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt',
        os.path.join(config.MODEL_DIR, 'whisper', 'tiny.pt'),
        'whisper'
    )
    download_file(
        'https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812',
        os.path.join(config.MODEL_DIR, 'face-parse-bisent', '79999_iter.pth'),
        'face parse'
    )
    download_file(
        'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        os.path.join(config.MODEL_DIR, 'face-parse-bisent', 'resnet18-5c106cde.pth'),
        'resnet'
    )
    print('Download over')
