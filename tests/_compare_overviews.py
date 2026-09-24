import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Compare rolling overviews across a soak run"
    )
    parser.add_argument("run_dir", help="Path to the soak run directory")
    args = parser.parse_args()
    run_dir = args.run_dir

    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory")
        return 1

    # Load all rolling overviews
    rolling = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.startswith("rolling_overview_") and fn.endswith(".json"):
            with open(os.path.join(run_dir, fn)) as f:
                data = json.load(f)
            t = int(fn.split("_")[2].replace(".json", ""))
            rolling.append((t, data))

    if not rolling:
        print("No rolling overviews found")
        return 1

    print("=== KEYS IN FIRST ROLLING OVERVIEW (turn {}) ===".format(rolling[0][0]))
    print(sorted(rolling[0][1].keys()))
    print()
    print("=== KEYS IN LAST ROLLING OVERVIEW (turn {}) ===".format(rolling[-1][0]))
    print(sorted(rolling[-1][1].keys()))

    # Show full content of the first rolling overview
    print("\n=== FIRST ROLLING OVERVIEW (turn {}) - FULL ===".format(rolling[0][0]))
    print(json.dumps(rolling[0][1], indent=2)[:2000])

    # Show summary scores across all rolling
    print("\n=== ROLLING OVERVIEW SCORES ===")
    scores = []
    for t, data in rolling:
        score = data.get("overall_score")
        scores.append((t, score))
        print("turn {:>3}: overall={}".format(t, score))

    valid_scores = [s for _, s in scores if s is not None]
    if valid_scores:
        print("\nValid scores: {}/{}".format(len(valid_scores), len(scores)))
        print("Average: {:.2f}".format(sum(valid_scores) / len(valid_scores)))
        print("Min: {}, Max: {}".format(min(valid_scores), max(valid_scores)))

    # Final overview
    final_path = os.path.join(run_dir, "final_overview.json")
    if os.path.exists(final_path):
        print("\n\n=== FINAL OVERVIEW ===")
        with open(final_path) as f:
            final = json.load(f)
        print("Keys:", sorted(final.keys()))
        print("overall_score: {}".format(final.get("overall_score")))
        print(
            "summary (first 200 chars): {}...".format(
                final.get("summary", "")[:200]
            )
        )
        print(
            "Final overview judge_records: {}".format(
                len(final.get("judge_records", []))
            )
        )
        print(
            "Last rolling (turn {}) judge_records: {}".format(
                rolling[-1][0],
                len(rolling[-1][1].get("judge_records", [])),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())