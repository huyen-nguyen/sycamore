"""
Step 2: Parse the ATLAS.ti export into a structured JSON evidence store.

Output: data/evidence.json
  [
    {
      "participant": "P2",
      "document": "P2_transcript",
      "quotation": "I use IGV the most...",
      "codes": ["TOOLS & LANGUAGES: IGV"],
      "code_categories": ["TOOLS & LANGUAGES"],
      "comment": ""
    }, ...
  ]

Chat history documents are excluded (sparse, non-interview content).
"""

import json
import os
import re
import openpyxl

ATLAS_FILE = "coding_info/Authoring_Genome-mapped_Data_Visualziations_Atlasti_Export__March2024.xlsx"
OUTPUT_FILE = "data/evidence.json"

PARTICIPANT_RE = re.compile(r"P(\d+)", re.IGNORECASE)


def extract_participant(document_name):
    m = PARTICIPANT_RE.search(document_name or "")
    return f"P{m.group(1)}" if m else None


def is_chat_history(document_name):
    return "chat_history" in (document_name or "").lower()


def extract_code_categories(codes_str):
    if not codes_str:
        return []
    categories = set()
    for code in codes_str.split(","):
        code = code.strip()
        if ":" in code:
            categories.add(code.split(":")[0].strip())
        else:
            categories.add(code)
    return sorted(categories)


def parse_evidence():
    wb = openpyxl.load_workbook(ATLAS_FILE)
    ws = wb.active

    records = []
    skipped_chat = 0
    skipped_no_participant = 0

    for row in ws.iter_rows(values_only=True, min_row=2):
        document, quotation, codes_raw, comment = row

        if is_chat_history(document):
            skipped_chat += 1
            continue

        participant = extract_participant(document)
        if not participant:
            skipped_no_participant += 1
            continue

        codes = [c.strip() for c in (codes_raw or "").split(",") if c.strip()]
        categories = extract_code_categories(codes_raw)

        records.append({
            "participant": participant,
            "document": document,
            "quotation": (quotation or "").replace("\n\t", " ").replace("\n", " ").strip(),
            "codes": codes,
            "code_categories": categories,
            "comment": (comment or "").strip(),
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=2)

    participants = sorted(set(r["participant"] for r in records))
    all_categories = sorted(set(cat for r in records for cat in r["code_categories"]))

    print(f"Parsed {len(records)} evidence records from {len(participants)} participants")
    print(f"Skipped: {skipped_chat} chat history rows, {skipped_no_participant} unidentifiable rows")
    print(f"\nParticipants in evidence: {participants}")
    print(f"\nAll code categories ({len(all_categories)}):")
    for cat in all_categories:
        print(f"  {cat}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_evidence()
