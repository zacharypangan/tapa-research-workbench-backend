"""Versioned project templates; the complete selected definition is pinned per run."""
from pathlib import Path
from .store import read_json

ONTOLOGY_TEMPLATES = [
    {"id": "general", "label": "General research", "description": "Documents, people, places, activities and evidence-grounded concepts."},
    {"id": "tapa", "label": "Tapa and archaeology", "description": "Barkcloth manufacture, tools, production cases and archaeological evidence."},
]


def ontology_template(template_id: str) -> dict:
    names = {"general": "default_general_ontology.json", "tapa": "default_ontology.json"}
    if template_id not in names:
        raise ValueError("Unknown ontology template; use general or tapa")
    return read_json(Path(__file__).with_name(names[template_id]))
