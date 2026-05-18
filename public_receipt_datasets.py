"""
Public datasets commonly used to benchmark receipt / document OCR (for evaluation or separate ML pipelines).

This project’s default path is Tesseract + rule-based parsing (`receipt_scanner.py`), not fine-tuning
those datasets here. Use them to measure accuracy or to train an external model, then plug in results.

Each entry: name, home page or paper, typical use.
"""

PUBLIC_RECEIPT_DATASETS: list[dict[str, str]] = [
    {
        "name": "CORD",
        "url": "https://github.com/clovaai/cord",
        "notes": "Receipt images + JSON key-value annotations (research benchmark).",
    },
    {
        "name": "SROIE (ICDAR2019)",
        "url": "https://rrc.cvc.uab.es/?ch=13",
        "notes": "Scanned receipts; key information extraction task.",
    },
    {
        "name": "FUNSD",
        "url": "https://guillaumejaume.github.io/FUNSD/",
        "notes": "Form understanding in noisy scans (not only receipts).",
    },
    {
        "name": "DocVQA",
        "url": "https://rrc.cvc.uab.es/?ch=17",
        "notes": "Document visual question answering; includes varied layouts.",
    },
]
