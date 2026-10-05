from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/processed/mist/benchmark_100")

RANKED_PATH = ROOT / "mist_multiformula_ranked_candidates.parquet"
MANIFEST_PATH = ROOT / "manifest.parquet"


SCORING_RULES = {
    "mist_only": 0.000,
    "mass_005": 0.005,
    "mass_010": 0.010,
    "mass_020": 0.020,
}


def reciprocal_rank_at_25(rank):
    if rank is None or rank > 25:
        return 0.0
    return 1.0 / rank


def evaluate(ranked, manifest, penalty):
    work = ranked.copy()

    work["score"] = (
        work["mist_cosine"]
        - penalty * work["formula_mass_error_ppm"].abs()
    )

    work = work.sort_values(
        [
            "query_inchikey14",
            "score",
            "candidate_inchikey14",
        ],
        ascending=[True, False, True],
        kind="stable",
    )

    work["new_rank"] = (
        work.groupby("query_inchikey14")
        .cumcount()
        + 1
    )

    rows = []

    for q in manifest.itertuples(index=False):
        query = work[
            work["query_inchikey14"]
            == q.query_inchikey14
        ]

        truth = query[
            query["candidate_inchikey14"]
            == q.query_inchikey14
        ]

        if truth.empty:
            rank = None
        else:
            rank = int(truth["new_rank"].min())

        rows.append(
            {
                "query_inchikey14": q.query_inchikey14,
                "rank": rank,
                "rr25": reciprocal_rank_at_25(rank),
                "top1": rank == 1,
                "top5": rank is not None and rank <= 5,
                "top10": rank is not None and rank <= 10,
                "top25": rank is not None and rank <= 25,
            }
        )

    return pd.DataFrame(rows)


def main():
    ranked = pd.read_parquet(RANKED_PATH)
    manifest = pd.read_parquet(MANIFEST_PATH)

    print("Multi-formula MIST + mass penalty")
    print("---------------------------------")

    results_by_rule = {}

    for name, penalty in SCORING_RULES.items():
        result = evaluate(
            ranked,
            manifest,
            penalty,
        )

        results_by_rule[name] = result

        found = result["rank"].notna()

        print(f"\n{name}")
        print(f"  penalty/ppm: {penalty:.3f}")
        print(
            f"  MRR@25: "
            f"{result['rr25'].mean():.4f}"
        )
        print(
            f"  Top1: "
            f"{result['top1'].mean() * 100:.2f}%"
        )
        print(
            f"  Top5: "
            f"{result['top5'].mean() * 100:.2f}%"
        )
        print(
            f"  Top10: "
            f"{result['top10'].mean() * 100:.2f}%"
        )
        print(
            f"  Top25: "
            f"{result['top25'].mean() * 100:.2f}%"
        )

        if found.any():
            print(
                f"  Median truth rank: "
                f"{result.loc[found, 'rank'].median():.1f}"
            )

    #
    # Compare each mass-penalized rule against MIST-only on the
    # individual queries.
    #
    baseline = results_by_rule["mist_only"].set_index(
        "query_inchikey14"
    )

    print("\nRank changes vs MIST-only")
    print("-------------------------")

    for name in [
        "mass_005",
        "mass_010",
        "mass_020",
    ]:
        current = results_by_rule[name].set_index(
            "query_inchikey14"
        )

        paired = baseline[["rank"]].join(
            current[["rank"]],
            lsuffix="_baseline",
            rsuffix="_new",
        )

        paired = paired[
            paired["rank_baseline"].notna()
            & paired["rank_new"].notna()
        ]

        delta = (
            paired["rank_new"]
            - paired["rank_baseline"]
        )

        print(f"\n{name}")
        print("  improved:", int((delta < 0).sum()))
        print("  unchanged:", int((delta == 0).sum()))
        print("  worsened:", int((delta > 0).sum()))
        print(
            "  mean rank change:",
            f"{delta.mean():.3f}",
        )
        print(
            "  worst degradation:",
            int(delta.max()),
        )
        print(
            "  best improvement:",
            int(delta.min()),
        )


if __name__ == "__main__":
    main()