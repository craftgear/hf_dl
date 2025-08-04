#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from typing import Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading


def download_hf_directory(
    repo_id: str,
    directory: str,
    local_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
    max_workers: int = 2,
):
    """
    Recursively download a directory from the specified Hugging Face repository
    """
    try:
        # If local directory is not specified, use repository name and directory name
        if local_dir is None:
            repo_name = repo_id.replace("/", "_")
            dir_name = directory.replace("/", "_")
            local_dir = f"{repo_name}_{dir_name}"

        # If output_dir is specified, create local_dir inside it
        if output_dir is not None:
            # Convert relative path to absolute path
            output_dir = os.path.abspath(os.path.expanduser(output_dir))
            local_dir = os.path.join(output_dir, local_dir)

        # Normalize directory path (remove leading/trailing slashes)
        directory = directory.strip("/")

        print(f"Downloading from {repo_id}/{directory}/ to {local_dir}")

        # Download only files under the specified directory
        allow_patterns = [f"{directory}/**/*", f"{directory}/*"]

        # Download the specified directory using snapshot_download
        # Limit concurrent downloads with max_workers
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            token=token,
            max_workers=max_workers,
        )

        print(f"Download completed! Files saved to: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"Error downloading files: {str(e)}")
        raise


def parse_hf_url(url: str) -> Tuple[str, str]:
    """
    Extract repo_id and directory from HuggingFace URL
    """
    # Parse URL
    parsed = urlparse(url)

    # For huggingface.co URLs
    if parsed.netloc == "huggingface.co":
        # Extract information from path (e.g., /username/repo-name/tree/main/directory/path)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid Hugging Face URL format")

        # Get repo_id
        repo_id = f"{path_parts[0]}/{path_parts[1]}"

        # Get directory path
        if len(path_parts) > 4 and path_parts[2] == "tree":
            # Treat everything after tree/branch_name/ as directory
            directory = "/".join(path_parts[4:])
        elif len(path_parts) > 2:
            # If tree/main is missing, treat 3rd element onwards as directory
            directory = "/".join(path_parts[2:])
        else:
            directory = ""

        return repo_id, directory
    else:
        raise ValueError("URL must be from huggingface.co")


def main():
    parser = argparse.ArgumentParser(
        description="Download files recursively from a Hugging Face repository directory",
        epilog="Example: python hf_download.py https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae",
    )
    parser.add_argument(
        "url",
        help="Hugging Face URL (e.g., 'https://huggingface.co/username/repo-name/tree/main/directory')",
    )
    parser.add_argument(
        "--local-dir",
        help="Specific local directory name to save files (default: auto-generated from URL)",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Base output directory where files will be saved (default: current directory)",
        default=None,
    )
    parser.add_argument(
        "--token", help="Hugging Face API token for private repositories", default=None
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent downloads (default: 2)",
        default=2,
    )

    args = parser.parse_args()

    try:
        repo_id, directory = parse_hf_url(args.url)

        if not directory:
            print("Error: Please specify a directory in the URL")
            print("Example: https://huggingface.co/username/repo-name/tree/main/models")
            return

        print(f"Repository: {repo_id}")
        print(f"Directory: {directory}")

        download_hf_directory(
            repo_id=repo_id,
            directory=directory,
            local_dir=args.local_dir,
            output_dir=args.output_dir,
            token=args.token,
            max_workers=args.max_workers,
        )
    except ValueError as e:
        print(f"Error: {str(e)}")
        parser.print_help()


if __name__ == "__main__":
    main()

