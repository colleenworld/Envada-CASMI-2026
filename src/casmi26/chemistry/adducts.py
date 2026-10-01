from dataclasses import dataclass


# Monoisotopic masses in Da.
H = 1.00782503223
PROTON = 1.007276466621
NA = 22.989218
K = 38.963158
NH4 = 18.033823
CL = 34.969402
FORMIC_ACID = 46.005479
H2O = 18.010565


@dataclass(frozen=True)
class Adduct:
    """
    Transformation between neutral molecular mass M and observed m/z.

    For the singly charged adducts used here:

        observed_mz = neutral_mass + mass_shift

    Therefore:

        neutral_mass = observed_mz - mass_shift
    """

    name: str
    mass_shift: float

    def neutral_mass(self, precursor_mz: float) -> float:
        return precursor_mz - self.mass_shift

    def precursor_mz(self, neutral_mass: float) -> float:
        return neutral_mass + self.mass_shift

ADDUCTS: dict[str, Adduct] = {
    "[M]-": Adduct(
        name="[M]-",
        mass_shift=0.0,
    ),
    "[M]+": Adduct(
        name="[M]+",
        mass_shift=0.0,
    ),
    "[M+H]+": Adduct(
        name="[M+H]+",
        mass_shift=PROTON,
    ),
    "[M-H]-": Adduct(
        name="[M-H]-",
        mass_shift=-PROTON,
    ),
    "[M+Na]+": Adduct(
        name="[M+Na]+",
        mass_shift=NA,
    ),
    "[M+K]+": Adduct(
        name="[M+K]+",
        mass_shift=K,
    ),
    "[M+NH4]+": Adduct(
        name="[M+NH4]+",
        mass_shift=NH4,
    ),
    "[M+CH2O2-H]-": Adduct(
        name="[M+CH2O2-H]-",
        mass_shift=FORMIC_ACID - PROTON,
    ),
    "[M+Cl]-": Adduct(
        name="[M+Cl]-",
        mass_shift=CL,
    ),
    "[M-H2O+H]+": Adduct(
        name="[M-H2O+H]+",
        mass_shift=PROTON - H2O,
    ),
    "[M-2H2O+H]+": Adduct(
        name="[M-2H2O+H]+",
        mass_shift=PROTON - (2 * H2O),
    ),
    "[M-H2O-H]-": Adduct(
        name="[M-H2O-H]-",
        mass_shift=-H2O - PROTON,
    ),
}


def neutral_mass(
    precursor_mz: float,
    adduct: str,
) -> float | None:
    definition = ADDUCTS.get(adduct)

    if definition is None:
        return None

    return definition.neutral_mass(
        precursor_mz
    )

def test_neutral_mass_for_negative_molecular_ion():
    assert neutral_mass(
        462.285,
        "[M]-",
    ) == 462.285