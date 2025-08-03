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
    指定されたHugging Faceリポジトリのディレクトリを再帰的にダウンロード
    """
    try:
        # ローカルディレクトリが指定されていない場合は、リポジトリ名とディレクトリ名を使用
        if local_dir is None:
            repo_name = repo_id.replace("/", "_")
            dir_name = directory.replace("/", "_")
            local_dir = f"{repo_name}_{dir_name}"

        # output_dirが指定されている場合、その中にlocal_dirを作成
        if output_dir is not None:
            # 相対パスを絶対パスに変換
            output_dir = os.path.abspath(os.path.expanduser(output_dir))
            local_dir = os.path.join(output_dir, local_dir)

        # ディレクトリパスの正規化（先頭・末尾のスラッシュを削除）
        directory = directory.strip("/")

        print(f"Downloading from {repo_id}/{directory}/ to {local_dir}")

        # 指定ディレクトリ配下のファイルのみをダウンロード
        allow_patterns = [f"{directory}/**/*", f"{directory}/*"]

        # snapshot_downloadを使用して指定ディレクトリをダウンロード
        # max_workersで同時ダウンロード数を制限
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
    HuggingFace URLからrepo_idとdirectoryを抽出
    """
    # URLのパース
    parsed = urlparse(url)

    # huggingface.co URLの場合
    if parsed.netloc == "huggingface.co":
        # パスから情報を抽出 (例: /username/repo-name/tree/main/directory/path)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid Hugging Face URL format")

        # repo_idを取得
        repo_id = f"{path_parts[0]}/{path_parts[1]}"

        # ディレクトリパスを取得
        if len(path_parts) > 4 and path_parts[2] == "tree":
            # tree/branch_name/以降をディレクトリとして扱う
            directory = "/".join(path_parts[4:])
        elif len(path_parts) > 2:
            # tree/mainがない場合は3番目以降をディレクトリとして扱う
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

