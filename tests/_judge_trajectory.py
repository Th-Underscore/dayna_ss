import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Show judge score trajectory across a soak run"
    )
    parser.add_argument("run_dir", help="Path to the soak run directory")
    args = parser.parse_args()
    run_dir = args.run_dir

    final_path = f"{run_dir}/final_overview.json"
    try:
        with open(final_path) as f:
            final = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {final_path}: {e}")
        return 1

    records = sorted(final.get("judge_records", []), key=lambda j: j["turn"])
    if not records:
        print("No judge records in final overview")
        return 1

    print(f"Total judge records: {len(records)}")
    print(f"{'turn':>5} {'style':>6} {'quality':>8} {'memory':>7}")
    for j in records:
        print(
            f"{j['turn']:>5} {j.get('style_score','?'):>6} "
            f"{j.get('quality_score','?'):>8} {j.get('memory_fidelity','?'):>7}"
        )

    # Compute averages over thirds
    if len(records) >= 3:
        third = len(records) // 3
        first_third = records[:third]
        mid_third = records[third:2*third]
        last_third = records[2*third:]

        def avg_field(recs, field):
            vals = [r.get(field) for r in recs
                    if isinstance(r.get(field), (int, float))]
            return sum(vals) / len(vals) if vals else None

        for label, recs in [("First third", first_third),
                            ("Middle third", mid_third),
                            ("Last third", last_third)]:
            print(f"\n{label} (turns {recs[0]['turn']}-{recs[-1]['turn']}):")
            print(f"  style avg: {avg_field(recs, 'style_score'):.2f}")
            print(f"  quality avg: {avg_field(recs, 'quality_score'):.2f}")
            print(f"  memory avg: {avg_field(recs, 'memory_fidelity'):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())