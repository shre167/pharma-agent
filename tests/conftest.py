import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharma_agent.config import load_settings  # noqa: E402
from pharma_agent.retrieval.knowledge_base import KnowledgeBase  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return load_settings(tmp_path)


@pytest.fixture
def synthetic_pdf(tmp_path) -> Path:
    """Two-column handbook-like PDF: bold 11pt uppercase titles, BOM and process labels.

    Page 1 left : ACETAMINOPHEN TABLETS (BOM + steps 1-2)
    Page 1 right: IBUPROFEN TABLETS (BOM + step 1)
    Page 2 left : continuation of IBUPROFEN (step 2) - no heading
    """
    path = tmp_path / "docs" / "handbook.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    def write(page, x, y, lines, bold=False, size=9):
        for line in lines:
            page.insert_text((x, y), line, fontsize=size, fontname="hebo" if bold else "helv")
            y += 14
        return y

    p1 = doc.new_page(width=612, height=792)
    y = write(p1, 50, 80, ["ACETAMINOPHEN TABLETS"], bold=True, size=11)
    write(p1, 50, y + 6, ["Bill of Materials", "Acetaminophen 500 mg", "Starch 50 mg",
                          "Manufacturing Directions", "1. Sift all ingredients.", "2. Blend for 10 minutes."])
    y = write(p1, 330, 80, ["IBUPROFEN TABLETS"], bold=True, size=11)
    write(p1, 330, y + 6, ["Bill of Materials", "Ibuprofen 200 mg", "Manufacturing Directions",
                           "1. Granulate with binder."])
    p2 = doc.new_page(width=612, height=792)
    write(p2, 50, 80, ["2. Compress into tablets."])
    doc.save(path)
    doc.close()
    return path


def make_chunk(cid, formulation, ctype, text, page=1, fid=None):
    fid = fid or f"fid_{formulation.split()[0].lower()}"
    return {"id": cid, "text": text, "type": ctype, "metadata": {
        "source": "handbook.pdf", "page": page, "pages": str(page), "formulation_id": fid,
        "formulation_name": formulation, "chunk_type": ctype, "active_ingredient": "x",
        "dosage_form": "tablets"}}


class FakeStore:
    """Stands in for FAISS: returns a fixed dense ranking of chunk ids."""

    def __init__(self, ordered_ids):
        self.ordered_ids = ordered_ids

    def similarity_search(self, query, k=4):
        return [SimpleNamespace(metadata={"chunk_id": cid}) for cid in self.ordered_ids[:k]]


@pytest.fixture
def small_kb():
    chunks = [
        make_chunk("c1", "ACETAMINOPHEN TABLETS", "BOM", "BILL OF MATERIALS acetaminophen 500 mg starch"),
        make_chunk("c2", "ACETAMINOPHEN TABLETS", "PROCESS", "MANUFACTURING PROCESS sift blend compress acetaminophen", page=2),
        make_chunk("c3", "IBUPROFEN TABLETS", "BOM", "BILL OF MATERIALS ibuprofen 200 mg"),
        make_chunk("c4", "IBUPROFEN TABLETS", "PROCESS", "MANUFACTURING PROCESS granulate ibuprofen", page=3),
    ]
    return KnowledgeBase(chunks, FakeStore(["c3", "c1", "c4", "c2"]))
