from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors

from casmi26.chemistry.formula import monoisotopic_mass
from casmi26.pubchem.mass_tier import PubChemMassTier


CANDIDATE_COLUMNS = [
    "candidate_formula",
    "candidate_inchikey14",
    "candidate_smiles",
    "candidate_source",
]


def load_mass_tier_candidates(
    formulas: set[str] | list[str],
    asset_root: Path,
    *,
    ppm: float = 10.0,
    excluded_structure_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    requested = sorted(
        {
            str(formula)
            for formula in formulas
            if str(formula)
        }
    )

    if not requested:
        return (
            pd.DataFrame(columns=CANDIDATE_COLUMNS),
            {
                "formulas": 0,
                "window_candidates": 0,
                "valid_smiles": 0,
                "formula_matches": 0,
                "unique_candidates": 0,
                "excluded_candidates": 0,
                "unsupported_formulas": 0,
            },
        )

    excluded = (
        excluded_structure_keys
        if excluded_structure_keys is not None
        else set()
    )

    tier = PubChemMassTier.load(
        Path(asset_root)
    )

    rows = []
    seen: set[tuple[str, str]] = set()

    stats = {
        "formulas": len(requested),
        "window_candidates": 0,
        "valid_smiles": 0,
        "formula_matches": 0,
        "unique_candidates": 0,
        "excluded_candidates": 0,
        "unsupported_formulas": 0,
    }

    for number, formula in enumerate(
        requested,
        start=1,
    ):
        try:
            target_mass = float(
                monoisotopic_mass(formula)
            )
        except ValueError:
            stats["unsupported_formulas"] += 1
            continue

        window_size = tier.window_size(
            target_mass,
            ppm=ppm,
        )
        stats["window_candidates"] += window_size

        formula_matches = 0

        for smiles in tier.iter_smiles_window(
            target_mass,
            ppm=ppm,
        ):
            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:
                continue

            stats["valid_smiles"] += 1

            try:
                candidate_formula = (
                    rdMolDescriptors.CalcMolFormula(
                        mol
                    )
                )
            except Exception:
                continue

            if candidate_formula != formula:
                continue

            formula_matches += 1
            stats["formula_matches"] += 1

            try:
                full_inchikey = inchi.MolToInchiKey(
                    mol
                )
            except Exception:
                continue

            if not full_inchikey:
                continue

            inchikey14 = str(
                full_inchikey
            ).split(
                "-"
            )[0]

            if not inchikey14:
                continue

            if inchikey14 in excluded:
                stats["excluded_candidates"] += 1
                continue

            pair = (
                formula,
                inchikey14,
            )

            if pair in seen:
                continue

            seen.add(
                pair
            )

            rows.append(
                {
                    "candidate_formula": formula,
                    "candidate_inchikey14": inchikey14,
                    "candidate_smiles": Chem.MolToSmiles(
                        mol,
                        canonical=True,
                        isomericSmiles=True,
                    ),
                    "candidate_source": "pubchem_mass_tier",
                }
            )

        if (
            number % 50 == 0
            or number == len(requested)
        ):
            print(
                f"  mass-tier formulas {number:,}/{len(requested):,} "
                f"| cumulative window candidates="
                f"{stats['window_candidates']:,} "
                f"| unique structures={len(rows):,}"
            )

    stats["unique_candidates"] = len(rows)

    result = pd.DataFrame(
        rows,
        columns=CANDIDATE_COLUMNS,
    )

    return result, stats
