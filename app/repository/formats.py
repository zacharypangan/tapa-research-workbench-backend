"""Supported upload formats and bounded DOCX reading without extra dependencies."""

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".vtt", ".srt", ".html", ".htm", ".xml"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | {".pdf", ".pptx", ".docx", ".rtf"}


def supported_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def docx_parts(file_path: str) -> list[tuple[str, bytes]]:
    with ZipFile(file_path) as archive:
        parts = [part for part in archive.infolist() if part.filename == "word/document.xml" or part.filename.startswith("word/media/")]
        if sum(part.file_size for part in parts) > 50 * 1024 * 1024:
            raise ValueError("DOCX contents exceed the extraction size limit.")
        return [(part.filename, archive.read(part)) for part in parts]


def docx_text(file_path: str) -> str:
    xml = next((data for name, data in docx_parts(file_path) if name == "word/document.xml"), None)
    if xml is None:
        raise ValueError("DOCX document text is missing.")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return "\n\n".join("".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)) for paragraph in root.findall(".//w:p", namespace))
