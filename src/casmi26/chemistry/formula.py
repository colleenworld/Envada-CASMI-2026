import re
from collections import defaultdict


MONOISOTOPIC_MASSES: dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.904468,
    "B": 11.00930536,
    "Si": 27.976926535,
    "Se": 79.9165218,
}


FORMULA_PATTERN = re.compile(
    r"([A-Z][a-z]?)(\d*)"
)


def parse_formula(
    formula: str,
) -> dict[str, int]:
    """
    Parse a simple molecular formula such as C20H30O2.

    Raises ValueError for malformed formulas or unsupported
    elements.
    """
    if not formula:
        raise ValueError(
            "Formula must not be empty."
        )

    composition: dict[str, int] = defaultdict(
        int
    )

    position = 0

    for match in FORMULA_PATTERN.finditer(
        formula
    ):
        if match.start() != position:
            raise ValueError(
                f"Unsupported molecular formula: {formula}"
            )

        element = match.group(1)

        if element not in MONOISOTOPIC_MASSES:
            raise ValueError(
                f"Unsupported element {element!r} "
                f"in formula {formula!r}"
            )

        count_text = match.group(2)

        count = (
            int(count_text)
            if count_text
            else 1
        )

        if count <= 0:
            raise ValueError(
                f"Invalid element count in {formula!r}"
            )

        composition[element] += count

        position = match.end()

    if (
        position != len(formula)
        or not composition
    ):
        raise ValueError(
            f"Unsupported molecular formula: {formula}"
        )

    return dict(
        composition
    )


def monoisotopic_mass(
    formula: str,
) -> float:
    """
    Calculate neutral monoisotopic mass from a molecular formula.
    """
    composition = parse_formula(
        formula
    )

    return sum(
        MONOISOTOPIC_MASSES[element]
        * count
        for element, count
        in composition.items()
    )


def mass_error_da(
    observed_mass: float,
    theoretical_mass: float,
) -> float:
    return (
        observed_mass
        - theoretical_mass
    )


def mass_error_ppm(
    observed_mass: float,
    theoretical_mass: float,
) -> float:
    if theoretical_mass == 0:
        raise ValueError(
            "Theoretical mass must be non-zero."
        )

    return (
        mass_error_da(
            observed_mass,
            theoretical_mass,
        )
        / theoretical_mass
        * 1_000_000.0
    )