#!/usr/bin/env python3
"""Plot a simple histogram of J values from a LaTeX table."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


TEXTCOLOR_RE = re.compile(r"\\textcolor\{[^}]+\}\{([^}]+)\}")


def parse_j_values(tex_path: Path) -> list[float]:
    values = []

    with tex_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            parts = [part.strip() for part in line.rstrip("\\").split("&")]
            if len(parts) < 3:
                continue

            j_text = parts[2]
            match = TEXTCOLOR_RE.search(j_text)
            if match:
                j_text = match.group(1)

            try:
                values.append(float(j_text))
            except ValueError:
                continue

    return values


def plot_histogram(j_values: list[float], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(j_values, bins="auto", edgecolor="black")
    plt.xlabel("J (meV)")
    plt.ylabel("Count")
    plt.title("Histogram of J values")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    tex_path = Path(
        "latex_tables/main_conf/slow_lowT_lowest_forces/all_Js_single_molecules.tex"
    )
    
    output_path = tex_path.with_name("all_Js_single_molecules_histogram.png")

    j_values = parse_j_values(tex_path)
    print(f"Found {len(j_values)} J values in {tex_path}")
    if not j_values:
        raise SystemExit("No J values found")

    plot_histogram(j_values, output_path)
    print(f"Saved histogram to {output_path}")