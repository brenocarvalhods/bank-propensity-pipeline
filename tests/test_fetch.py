"""Unit tests for the dataset fetch helpers, using in-memory zip archives (no network)."""

import io
import zipfile

import pytest

from bank_propensity_pipeline.data.fetch import (
    INNER_ARCHIVE,
    TARGET_FILE,
    extract_dataset,
    extract_member,
    save_bytes,
)


def make_zip(files: dict[str, bytes]) -> bytes:
    """Build a zip archive in memory with the given file names and contents."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_extract_member_ignores_macos_metadata() -> None:
    archive = make_zip({"__MACOSX/data.csv": b"junk", "folder/data.csv": b"real"})
    assert extract_member(archive, "data.csv") == b"real"


def test_extract_member_raises_when_file_is_missing() -> None:
    with pytest.raises(FileNotFoundError):
        extract_member(make_zip({"other.txt": b"x"}), "data.csv")


def test_extract_dataset_reads_csv_inside_nested_archive() -> None:
    inner = make_zip({f"bank-additional/{TARGET_FILE}": b"age;y\n30;no\n"})
    outer = make_zip({INNER_ARCHIVE: inner, "bank.zip": b"ignored"})
    assert extract_dataset(outer) == b"age;y\n30;no\n"


def test_save_bytes_creates_parent_folders(tmp_path) -> None:
    path = tmp_path / "raw" / "file.csv"
    save_bytes(b"content", path)
    assert path.read_bytes() == b"content"
