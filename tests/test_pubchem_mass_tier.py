from pathlib import Path

import numpy as np

from casmi26.chemistry.formula import monoisotopic_mass
from casmi26.pubchem.mass_tier import PubChemMassTier
from casmi26.pubchem.mass_tier_candidates import (
    load_mass_tier_candidates,
)


def write_mass_tier(
    root: Path,
    entries: list[tuple[str, str]],
) -> None:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = sorted(
        [
            (
                monoisotopic_mass(formula),
                smiles,
            )
            for formula, smiles in entries
        ],
        key=lambda item: item[0],
    )

    masses = np.asarray(
        [
            mass
            for mass, _ in ordered
        ],
        dtype=np.float64,
    )

    offsets = [0]
    encoded = bytearray()

    for _, smiles in ordered:
        value = smiles.encode("ascii")
        encoded.extend(value)
        offsets.append(len(encoded))

    np.save(
        root / "pubchem_tier__pc_mass.npy",
        masses,
    )
    np.save(
        root / "pubchem_tier__pc_off.npy",
        np.asarray(
            offsets,
            dtype=np.uint64,
        ),
    )
    np.save(
        root / "pubchem_tier__pc_smiles.npy",
        np.frombuffer(
            bytes(encoded),
            dtype=np.uint8,
        ),
    )


def test_mass_tier_iterates_only_the_requested_mass_window(
    tmp_path: Path,
) -> None:
    write_mass_tier(
        tmp_path,
        [
            ("C2H6O", "CCO"),
            ("C3H8O", "CCCO"),
        ],
    )

    tier = PubChemMassTier.load(
        tmp_path
    )

    assert tier.smiles_window(
        monoisotopic_mass("C2H6O"),
        ppm=10.0,
    ) == ["CCO"]


def test_mass_tier_candidates_are_formula_checked_and_keyed(
    tmp_path: Path,
) -> None:
    write_mass_tier(
        tmp_path,
        [
            ("C2H6O", "CCO"),
            ("C3H8O", "CCCO"),
        ],
    )

    candidates, stats = load_mass_tier_candidates(
        {
            "C2H6O",
            "C3H8O",
        },
        tmp_path,
    )

    assert set(
        candidates["candidate_formula"]
    ) == {
        "C2H6O",
        "C3H8O",
    }
    assert candidates[
        "candidate_inchikey14"
    ].str.len().eq(14).all()
    assert candidates[
        "candidate_source"
    ].eq("pubchem_mass_tier").all()
    assert stats["unique_candidates"] == 2


def test_mass_tier_candidates_respect_excluded_structure_keys(
    tmp_path: Path,
) -> None:
    write_mass_tier(
        tmp_path,
        [
            ("C2H6O", "CCO"),
            ("C3H8O", "CCCO"),
        ],
    )

    candidates, _ = load_mass_tier_candidates(
        {
            "C2H6O",
            "C3H8O",
        },
        tmp_path,
        excluded_structure_keys={
            "LFQSCWFLJHTTHZ",
        },
    )

    assert "LFQSCWFLJHTTHZ" not in set(
        candidates["candidate_inchikey14"]
    )
    assert candidates[
        "candidate_formula"
    ].tolist() == ["C3H8O"]
