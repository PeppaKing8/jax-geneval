from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load_scores(path: str):
    counts = defaultdict(lambda: [0, 0])
    total = [0, 0]
    with open(path) as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            tag = row["tag"]
            correct = int(bool(row["correct"]))
            counts[tag][0] += correct
            counts[tag][1] += 1
            total[0] += correct
            total[1] += 1
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    counts, total = load_scores(args.results)
    print(f"image_score {total[0]}/{total[1]} = {total[0] / total[1]:.6f}")
    task_scores = []
    for tag in sorted(counts):
        correct, count = counts[tag]
        score = correct / count if count else 0.0
        task_scores.append(score)
        print(f"{tag} {correct}/{count} = {score:.6f}")
    if task_scores:
        print(f"task_average = {sum(task_scores) / len(task_scores):.6f}")

    if args.compare:
        other_counts, other_total = load_scores(args.compare)
        print("")
        print(f"compare_image_score {other_total[0]}/{other_total[1]} = {other_total[0] / other_total[1]:.6f}")
        for tag in sorted(set(counts) | set(other_counts)):
            a = counts[tag]
            b = other_counts[tag]
            a_score = a[0] / a[1] if a[1] else 0.0
            b_score = b[0] / b[1] if b[1] else 0.0
            print(f"delta {tag}: {a_score - b_score:+.6f}")


if __name__ == "__main__":
    main()
