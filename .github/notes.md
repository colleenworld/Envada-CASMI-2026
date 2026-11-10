## 2026-11-04: Contrastive ranking mass penalty

Raised the positive MIST contrastive ranker's cross-formula mass-error
penalty from `0.020` to `0.075`. A joint sweep on the frozen, structure-
disjoint v1 and v2 validation sets reproduced the production behavior:
collapse each molecule/structure pair to its best formula-hypothesis score,
then rank structures.

| Validation set | MRR@25 at 0.020 | MRR@25 at 0.075 | Top-25 at 0.020 | Top-25 at 0.075 |
|---|---:|---:|---:|---:|
| v1 (500 queries) | 0.5439 | 0.5509 | 79.8% | 80.0% |
| v2 (500 queries) | 0.5494 | 0.5622 | 86.0% | 86.6% |

This is an offline validation improvement, not a confirmed Kaggle leaderboard
gain; it still needs a new Kaggle inference/submission to measure the hidden
test score.

## Phase 2: Offline MSBuddy data packaging

The Phase 2 Kaggle run reached MSBuddy initialization but failed because
MSBuddy attempted to download its reference databases while Kaggle internet
access was disabled. The inference now supplies explicit paths for MSBuddy's
common database, formula database, and ML model. The asset refresh script
packages and checks these files under `project/data/external/msbuddy/`, and
the runner fails preflight if any file is absent.

The packaged files total about 824 MiB. Their SHA-256 hashes matched the
installed MSBuddy 0.3.16 data files, and an offline-path smoke test returned
formula candidates. This resolves the identified network dependency, but a
new Kaggle run is still required to confirm a valid submission.

## 2026-11-04: Retrieval + fusion upgrade (no new dependencies)

Started a no-new-dependency scoring upgrade focused on candidate coverage and
rank fusion:

1. `scripts/rank_test_negative_retrieval.py` now supports
   `CASMI26_RETRIEVAL_INCLUDE_POSITIVE_MOLECULES=1` to rank retrieval
   candidates for all molecules, not only molecules missing positive-MIST
   output.
2. `scripts/build_submission.py` now supports
   `CASMI26_SUBMISSION_FUSION_MODE=rrf` and fuses positive/fallback candidate
   lists using reciprocal-rank fusion with configurable `k` and branch
   weights.
3. `scripts/run_kaggle_phase2_inference.py` enables this experiment path by
   setting:
   - `CASMI26_RETRIEVAL_INCLUDE_POSITIVE_MOLECULES=1`
   - `CASMI26_SUBMISSION_FUSION_MODE=rrf`
   - `CASMI26_SUBMISSION_RRF_K=60.0`
   - equal positive/fallback fusion weights

Rationale: this directly targets a known failure mode where the positive branch
contains candidates but not the top-ranked truth, while the spectral retrieval
branch can provide complementary structures for the same molecule.

## 2026-11-04: Adopted existing LightGBM inference path

Per request to avoid reinvention, we switched to reusing the existing
learned-ranker implementation already present in the Kaggle project snapshot.

Added these scripts to the main repository `scripts/` from the existing
working path:

- `build_generated_test_formula_shortlist.py`
- `apply_generated_test_formula_hypotheses.py`
- `prepare_test_guard5_candidates.py`
- `score_test_guard5_spectrum_fingerprint.py`
- `build_test_learned_ranker_features.py`
- `rank_test_learned_ranker.py`
- `run_kaggle_guard5_inference.py`

Updated `scripts/refresh_kaggle_inference_assets.sh` so Kaggle assets now also
refresh and preflight the learned-ranker prerequisites:

- guard-5 script set above
- `data/processed/pubchem_formula_index/` candidate buckets
- spectrum-fingerprint model checkpoint
- LightGBM ranker model file
- `coconut_full.parquet`

No new dependency installation was performed in this step.

## 2026-11-04: Morgan deprecation warning cleanup

Observed heavy RDKit warning spam: `DEPRECATION WARNING: please use MorganGenerator`.

Changes:

1. `scripts/prepare_test_guard5_candidates.py` now uses
   `rdFingerprintGenerator.GetMorganGenerator(...).GetFingerprint(...)`
   instead of `AllChem.GetMorganFingerprintAsBitVect`.
2. `scripts/score_test_guard5_spectrum_fingerprint.py` now uses
   `MorganGenerator` for candidate fingerprints.
3. `scripts/build_test_contrastive_candidates.py` now suppresses
   `rdApp.warning` before calling MIST's retrieval HDF builder, because that
   warning stream is emitted inside MIST internals during bulk fingerprint
   generation.

This removes noisy warning output and avoids deprecated-call overhead in our
code paths without changing ranking semantics.

## 2026-11-04: Fast iteration mode vs final submission mode

Added an explicit run-mode split so long-running heavy retrieval/fusion only
runs when requested for final submissions.

1. `scripts/run_kaggle_phase2_inference.py` now accepts
   `--final-submission`.
2. Default mode (no flag) is `iteration-fast`:
   - `CASMI26_RETRIEVAL_INCLUDE_POSITIVE_MOLECULES=0`
   - `CASMI26_SUBMISSION_FUSION_MODE=strict`
3. Final mode (`--final-submission`) is heavy:
   - `CASMI26_RETRIEVAL_INCLUDE_POSITIVE_MOLECULES=1`
   - `CASMI26_SUBMISSION_FUSION_MODE=rrf`
4. Notebook entrypoint now has `FINAL_SUBMISSION_MODE = False` by default and
   only adds `--final-submission` when explicitly enabled.

This keeps normal Kaggle iteration runs in the previous fast regime while
reserving hour-scale behavior for intentional final submissions.

## 2026-11-04: Additional iteration runtime guardrails

Iteration mode was still running too long on hidden-test Kaggle runs, so we
added stricter bounded defaults for non-final runs:

1. `scripts/run_kaggle_phase2_inference.py` now sets:
   - iteration mode: `MIST --top-k 200`
   - final mode: `MIST --top-k 500`
2. Added `CASMI26_MAX_STRUCTURES_PER_FORMULA` support in
   `scripts/build_test_contrastive_candidates.py`:
   - iteration mode sets cap to `250` structures per formula
   - final mode keeps uncapped (`0`)
3. Formula cap preferentially keeps training/COCONUT structures first and logs
   before/after candidate counts.

These guardrails are for fast iteration only; final-submission mode still
preserves the full heavy candidate/retrieval path.

## 2026-11-05: Kaggle failure root cause (`FEATURE_ID` KeyError)

Latest `casmi26-phase-2-mist-offline` run failed in MIST subformula assignment:

- `assign_subformulae.py` raised `KeyError: 'FEATURE_ID'`
- called via `--feature-id FEATURE_ID`

Our exported MGF records hypothesis IDs in `TITLE=...`, not a `FEATURE_ID`
field. Updated both the phase-2 runner and notebook fallback path to pass:

- `--feature-id TITLE`

This aligns MIST's identifier lookup with the actual MGF metadata.

## 2026-11-05: Enforced sub-20-minute test mode

Added explicit quick-test mode to keep routine Kaggle test runs bounded and
fast:

1. `scripts/run_kaggle_phase2_inference.py` now supports `--quick-test`.
2. Quick-test mode skips heavy MIST/MSBuddy stages and runs:
   - write empty positive ranking artifact
   - same-polarity retrieval ranking
   - submission assembly/validation
3. Notebook defaults now include `QUICK_TEST_MODE = True`, which adds
   `--quick-test` to the phase-2 runner command.

Final leaderboard runs remain available by disabling `QUICK_TEST_MODE` and
optionally enabling `FINAL_SUBMISSION_MODE`.

## 2026-11-05: Strict-mode candidate cap fix

Quick-test run failed with:
`ValueError: At least one molecule has more than 25 predictions.`

Root cause: `build_submission.py` strict fusion path was not capping
`submission_rank` to the Kaggle maximum of 25 after allowing larger
branch-level candidate pools.

Fix: strict fusion now enforces `submission_rank <= 25` before final
aggregation.

## 2026-11-05: Switched notebook defaults for scoring run

Set Kaggle notebook defaults to launch a scoring-oriented run:

- `FINAL_SUBMISSION_MODE = True`
- `QUICK_TEST_MODE = False`

This enables the full heavy final path by default for the next kernel push.

## 2026-11-05: Submission workflow clarified in project.md

Added an explicit project rule that competition submissions must be made via the
Kaggle notebook UI using **Submit to competition**. Documented that
`kaggle competitions submit` is not part of this repository's canonical workflow.

## 2026-11-04: RDKit explicit-valence log spam source

During Kaggle runs we observed repeated messages like:
`Explicit valence for atom # ... Br, 3, is greater than permitted`.

These originate from RDKit parsing invalid candidate SMILES during MIST HDF
construction. They are non-fatal (those SMILES are skipped) but created heavy
stderr spam.

Updated `scripts/build_test_contrastive_candidates.py` to suppress all RDKit
`rdApp.*` messages before invoking MIST's retrieval HDF builder, so this
non-actionable parser noise no longer floods Kaggle logs.

## 2026-11-06: Public-v4 fusion ablation and method screen

On the same 250-query private validation set, changing `FUSE_ALPHA` from 0.60
to 0.30 improved MRR@25 from 0.8406 to 0.8854 (+0.0447; paired-bootstrap
95% CI +0.0276 to +0.0640). Top-1 increased from 74.4% to 82.4%, Top-25
remained 98.0%, and 27 queries improved with none worsening. This is a
promising ablation, not evidence of a leaderboard gain.

The public-v4 kernel was first published as version 21, which was canceled
while queued, then republished unchanged as version 22. Kaggle reported
version 22 complete. The user reports a competition score of 0.384, down from
the prior 0.403. Thus the alpha-0.30 change did not transfer as hoped; because
the competition run is the only result on the hidden set, do not claim the
private validation improvement predicts leaderboard performance.

The alpha validation queries are not the available structure-disjoint
`enveda-np-examples` benchmark: the two sets have zero overlapping structures
(250 versus 246). The high validation MRR therefore must not be used as a
proxy for expected hidden-test performance. Full local v4 model/candidate
assets are not present, so the representative end-to-end comparison must use
the Kaggle asset environment.

Existing experiments help prioritize the next methods:

- Entropy-neighbor features raised MRR@25 from 0.2375 to 0.2538 on a separate
  420-query learned-ranker holdout (paired-bootstrap 95% CI +0.0059 to +0.0273).
  This is promising offline evidence, but it is not the public-v4 pipeline.
- Adding MS2DeepScore on that same ranker changed MRR by only +0.0001
  (95% CI -0.0067 to +0.0068; bootstrap support for a gain 51.5%), so it is
  not a current priority.
- On the 1,000-query source-coverage audit, PubChemLite matched 65 of 893
  truths already in the PubChem mass-tier set, but only 6 hits were absent
  from the existing public-v2 pool. Test it on the domain-matched benchmark
  before deciding whether its incremental coverage is useful.

Before another competition attempt, compare alpha 0.30 against the 0.60
reference on the same structure-disjoint benchmark with every other part of
the inference source held fixed. Then evaluate candidate-source and ranking
changes there. Reuse existing assets and dependencies first; add no library
without a measured offline gain.
