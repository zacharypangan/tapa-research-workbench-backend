import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STORAGE_ROOT = os.getenv(
    "REPOSITORY_STORAGE_ROOT",
    os.path.join(BASE_DIR, "storage", "repository"),
)
FILES_ROOT = os.path.join(STORAGE_ROOT, "files")
IMAGES_ROOT = os.path.join(STORAGE_ROOT, "images")
DB_PATH = os.path.join(STORAGE_ROOT, "repository.sqlite")
OLLAMA_REPOSITORY_BASE_URL = os.getenv(
    "REPOSITORY_OLLAMA_BASE_URL",
    os.getenv("OLLAMA_BASE_URL", ""),
).rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_RETRIEVAL_MODEL = os.getenv("OLLAMA_RETRIEVAL_MODEL", "llama3.1")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
EMBEDDING_TEXT_LIMIT = 2000
MULTIMODAL_METHOD_VERSION = "multimodal_evidence_v2_context"
IMAGE_INDEX_JOBS: dict[str, dict] = {}

SOURCE_TYPES = {
    "publication",
    "ethnography",
    "presentation",
    "dictionary",
    "anthropological_record",
    "workshop_material",
    "bibliography",
    "pdf",
    "slide",
    "other",
}

STATUSES = {
    "uploaded",
    "needs_metadata",
    "metadata_complete",
    "needs_review",
    "ready_for_text_extraction",
}

OBSERVATION_TYPES = {
    "term",
    "motif",
    "place",
    "material",
    "process",
    "other",
}
