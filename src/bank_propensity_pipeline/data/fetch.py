"""Download the Bank Marketing dataset from the UCI Machine Learning Repository."""

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

from bank_propensity_pipeline.config import PROJECT_ROOT, load_params, load_settings
from bank_propensity_pipeline.logging_config import setup_logging

logger = logging.getLogger(__name__)

TARGET_FILE = "bank-additional-full.csv"
INNER_ARCHIVE = "bank-additional.zip"


def download_archive(url: str) -> bytes:
    """Download a file from a URL and return its raw bytes."""
    logger.info("Downloading dataset from %s", url)
    with urlopen(url, timeout=60) as response:
        return response.read()


def extract_member(archive: bytes, suffix: str) -> bytes:
    """Return the content of the first archive member whose name ends with suffix."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        for name in zip_file.namelist():
            if name.endswith(suffix) and not name.startswith("__MACOSX"):
                return zip_file.read(name)
    raise FileNotFoundError(f"No member ending with '{suffix}' in archive")


def extract_dataset(archive: bytes) -> bytes:
    """Find the target CSV, looking inside the nested archive when needed."""
    try:
        return extract_member(archive, TARGET_FILE)
    except FileNotFoundError:
        inner_archive = extract_member(archive, INNER_ARCHIVE)
        return extract_member(inner_archive, TARGET_FILE)


def save_bytes(content: bytes, path: Path) -> None:
    """Write raw bytes to disk, creating parent folders when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    logger.info("Saved %s (%d bytes)", path, len(content))


def fetch_dataset(url: str, output_path: Path) -> None:
    """Download the UCI archive and save the full bank-additional CSV."""
    archive = download_archive(url)
    csv_content = extract_dataset(archive)
    save_bytes(csv_content, output_path)


def main() -> None:
    """Entry point used by the DVC fetch stage."""
    setup_logging(load_settings().log_level)
    params = load_params()
    fetch_dataset(params["data"]["source_url"], PROJECT_ROOT / params["data"]["raw_path"])


if __name__ == "__main__":
    main()
