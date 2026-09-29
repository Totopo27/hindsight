"""FinanceBench PDFs -> markdown (tables kept), one page marker per page. Cached."""
import sys
from multiprocessing import Pool
from pathlib import Path

import os

ROOT = Path.home() / ".cache/hindsight/kbbench-data"
SRC = Path(os.environ.get("PDF_SRC", ROOT / "fb/pdfs"))
OUT = Path(os.environ.get("MD_OUT", ROOT / "fb-md"))


def convert(pdf: Path) -> str:
    out = OUT / (pdf.stem + ".md")
    if out.exists():
        return f"skip {pdf.stem}"
    import pymupdf4llm
    try:
        pages = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, show_progress=False)
        text = "\n\n".join(f"<!-- page {i + 1} -->\n{p['text']}" for i, p in enumerate(pages))
    except Exception as e:  # a PDF the markdown path chokes on still gets plain text
        import pymupdf
        doc = pymupdf.open(pdf)
        text = "\n\n".join(f"<!-- page {i + 1} -->\n{pg.get_text('text', sort=True)}" for i, pg in enumerate(doc))
        print(f"fallback {pdf.stem}: {e}", flush=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.rename(out)
    return f"done {pdf.stem} {len(text)}"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    first = set((ROOT / "fb-docs.txt").read_text().split()) if "fb" in str(SRC) else set()
    pdfs = sorted(SRC.glob("*.pdf"), key=lambda p: (p.stem not in first, p.stem))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for i, msg in enumerate(pool.imap(convert, pdfs), 1):
            print(f"[{i}/{len(pdfs)}] {msg}", flush=True)
