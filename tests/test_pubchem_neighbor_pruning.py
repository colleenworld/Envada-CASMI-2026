import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from casmi26.pubchem.neighbor_pruning import (
    score_candidates,
)


def fp(smiles):
    mol = Chem.MolFromSmiles(
        smiles
    )

    return (
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=4096,
        )
    )


def test_neighbor_prior_prefers_structurally_similar_candidate():
    candidate_fps = [
        fp("CCO"),
        fp("c1ccccc1"),
    ]

    training_fps = {
        "NEIGHBOR": fp("CCCO"),
    }

    scores, usable = (
        score_candidates(
            candidate_fps,
            [
                (
                    "NEIGHBOR",
                    0.8,
                ),
            ],
            training_fps,
        )
    )

    assert usable == 1
    assert scores[0] > scores[1]


def test_neighbor_prior_uses_spectral_weight():
    candidate_fps = [
        fp("CCO"),
    ]

    training_fps = {
        "A": fp("CCO"),
        "B": fp("CCO"),
    }

    scores, usable = (
        score_candidates(
            candidate_fps,
            [
                ("A", 0.9),
                ("B", 0.2),
            ],
            training_fps,
        )
    )

    assert usable == 2
    assert np.isclose(
        scores[0],
        0.9,
    )
