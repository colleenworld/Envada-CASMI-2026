from collections.abc import Sequence

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def smiles_to_inchikey14(smiles: str) -> str | None:
    """Convert a SMILES prediction to its tautomer-canonical InChIKey14."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical = enumerator.Canonicalize(mol)

    inchikey = Chem.MolToInchiKey(canonical)

    return inchikey[:14]


def reciprocal_rank(
    truth_smiles: str,
    predictions: Sequence[str],
    k: int = 25,
) -> float:
    """
    Reciprocal rank of the first correct prediction.

    Returns zero if no prediction in the first k candidates matches.
    """

    truth_key = smiles_to_inchikey14(truth_smiles)

    if truth_key is None:
        raise ValueError(f"Invalid truth SMILES: {truth_smiles!r}")

    for rank, prediction in enumerate(predictions[:k], start=1):
        prediction_key = smiles_to_inchikey14(prediction)

        if prediction_key == truth_key:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    truths: Sequence[str],
    predictions: Sequence[Sequence[str]],
    k: int = 25,
) -> float:
    """Calculate MRR@k over molecules."""

    if len(truths) != len(predictions):
        raise ValueError(
            "truths and predictions must contain the same number of molecules"
        )

    if not truths:
        raise ValueError("Cannot calculate MRR for an empty dataset")

    scores = [
        reciprocal_rank(truth, candidates, k=k)
        for truth, candidates in zip(truths, predictions, strict=True)
    ]

    return sum(scores) / len(scores)