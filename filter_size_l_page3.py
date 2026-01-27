import csv


INPUT_PATH = "shipping_summary_regionals_v3.csv"
OUTPUT_PATH = "page3.csv"
SIZE_VALUE = "Size L"


def main() -> None:
    with open(INPUT_PATH, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if "PackageSize" not in fieldnames:
            raise ValueError("PackageSize column not found in input CSV.")

        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("PackageSize") == SIZE_VALUE:
                    writer.writerow(row)


if __name__ == "__main__":
    main()
