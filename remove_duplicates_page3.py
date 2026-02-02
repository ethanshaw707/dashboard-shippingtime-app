import csv
from pathlib import Path


def remove_duplicates(input_path: Path, output_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return 0

    header = rows[0]
    data = rows[1:]

    seen = set()
    deduped = []
    for row in data:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    removed = len(data) - len(deduped)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(deduped)

    return removed


if __name__ == "__main__":
    input_path = Path("page3.csv")
    output_path = Path("page3_deduped.csv")
    removed = remove_duplicates(input_path, output_path)
    print(f"Removed {removed} duplicate rows.")
