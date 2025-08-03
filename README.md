# HuggingFace Directory Downloader

A Python script to download specific directories from HuggingFace repositories with concurrent download control.

## Features

- Download specific directories from HuggingFace repositories
- Control concurrent downloads to prevent network overload
- Resume interrupted downloads
- Support for private repositories with API tokens
- Customizable output directory

## Requirements

```bash
pip install huggingface-hub
```

## Usage

### Basic Usage

Download a directory from a HuggingFace repository:

```bash
python hf_download.py https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae
```

### Command Line Arguments

- `url` - HuggingFace URL to the directory you want to download
- `--output-dir` - Base directory where files will be saved (default: current directory)
- `--local-dir` - Custom name for the downloaded folder (default: auto-generated from repo/directory name)
- `--max-workers` - Maximum number of concurrent downloads (default: 2)
- `--token` - HuggingFace API token for private repositories

### Examples

#### Download with limited concurrent connections
```bash
python hf_download.py https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae --max-workers 1
```

#### Download to specific output directory
```bash
python hf_download.py https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/safety_checker --output-dir ~/models
```

#### Download with custom folder name
```bash
python hf_download.py https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/text_encoder --output-dir ~/models --local-dir sd15-text-encoder
```

#### Download from private repository
```bash
python hf_download.py https://huggingface.co/private-repo/model/tree/main/weights --token YOUR_HF_TOKEN
```

#### Full example with all options
```bash
python hf_download.py https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet \
  --output-dir ~/sdxl-models \
  --local-dir sdxl-unet \
  --max-workers 3 \
  --token YOUR_HF_TOKEN
```

## How it works

1. Parses the HuggingFace URL to extract repository ID and directory path
2. Uses `huggingface_hub.snapshot_download()` with pattern matching to download only files in the specified directory
3. Supports resuming interrupted downloads automatically
4. Limits concurrent downloads to prevent overwhelming your network connection

## Notes

- The script will create the output directory if it doesn't exist
- Downloads are resumed automatically if interrupted
- Files are saved with their original directory structure preserved
- Default concurrent downloads is set to 2 to balance speed and stability
- Directory names with spaces are fully supported (e.g., `--output-dir "My Models"` or `--output-dir My\ Models`)