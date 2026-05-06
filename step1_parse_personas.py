"""
Step 1: Parse personas.xlsx into a structured JSON file.

Output: data/personas.json
  {
    "Biologists": {
      "participants": ["P20", "P17", ...],
      "members": {
        "P20": {
          "position": "RA + MS",
          "org": "U",
          "skills": {"genomics": 1, "data_prep": 2, "programming": 2, "vis": 1},
          "bio_persona": ["User"],
          "ds_persona": ["E"],
          "focus": "Bio",
          "automation": "Low",
          "audience": "Self"
        }, ...
      }
    }, ...
  }

Excludes "Vis. Experts" (n=1, data too sparse).
"""

import json
import os
import openpyxl

PERSONAS_FILE = "personas.xlsx"
OUTPUT_FILE = "data/personas.json"
EXCLUDE_GROUPS = {"Vis. Experts"}

BIO_PERSONA_COLS = {9: "User", 10: "Scientist", 11: "Engineer"}
DS_PERSONA_COLS = {
    12: "DSh/DSt", 13: "D Eng", 14: "*Eng",
    15: "G", 16: "RS", 17: "TA", 18: "M", 19: "E"
}


def parse_personas():
    wb = openpyxl.load_workbook(PERSONAS_FILE)
    ws = wb.active

    personas = {}
    current_group = None

    # Rows 1-3 are headers; data starts at row 4 (index 3)
    for row in list(ws.iter_rows(values_only=True))[3:]:
        group_cell = row[0]
        participant_id = row[1]

        if group_cell is not None:
            current_group = group_cell.strip()

        if current_group in EXCLUDE_GROUPS:
            continue

        if not participant_id:
            continue

        pid = participant_id.strip()

        member = {
            "position": row[3],
            "org": row[4],
            "skills": {
                "genomics": row[5],
                "data_prep": row[6],
                "programming": row[7],
                "vis": row[8],
            },
            "bio_persona": [
                label for col, label in BIO_PERSONA_COLS.items()
                if row[col] and str(row[col]).strip()
            ],
            "ds_persona": [
                label for col, label in DS_PERSONA_COLS.items()
                if row[col] and str(row[col]).strip()
            ],
            "focus": row[20],
            "automation": row[21],
            "audience": row[22],
        }

        if current_group not in personas:
            personas[current_group] = {"participants": [], "members": {}}

        personas[current_group]["participants"].append(pid)
        personas[current_group]["members"][pid] = member

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(personas, f, indent=2)

    print(f"Parsed {sum(len(v['participants']) for v in personas.values())} participants "
          f"across {len(personas)} persona groups:")
    for group, data in personas.items():
        print(f"  {group}: {data['participants']}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_personas()
