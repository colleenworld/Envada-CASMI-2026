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
    name: str
    mass_shift: float
    molecule_count: int = 1
    charge: int = 1

    def neutral_mass(
        self,
        precursor_mz: float,
    ) -> float:
        return (
            precursor_mz * abs(self.charge)
            - self.mass_shift
        ) / self.molecule_count

    def precursor_mz(
        self,
        neutral_mass: float,
    ) -> float:
        return (
            self.molecule_count * neutral_mass
            + self.mass_shift
        ) / abs(self.charge)

ADDUCTS: dict[str, Adduct] = {
    "[M]+": Adduct(
        "[M]+",
        0.0,
    ),
    "[M]-": Adduct(
        "[M]-",
        0.0,
        charge=-1,
    ),

    "[M+H]+": Adduct(
        "[M+H]+",
        PROTON,
    ),
    "[M-H]-": Adduct(
        "[M-H]-",
        -PROTON,
        charge=-1,
    ),

    "[M+Na]+": Adduct(
        "[M+Na]+",
        NA,
    ),
    "[M+K]+": Adduct(
        "[M+K]+",
        K,
    ),
    "[M+NH4]+": Adduct(
        "[M+NH4]+",
        NH4,
    ),

    "[M+CH2O2-H]-": Adduct(
        "[M+CH2O2-H]-",
        FORMIC_ACID - PROTON,
        charge=-1,
    ),
    "[M+Cl]-": Adduct(
        "[M+Cl]-",
        CL,
        charge=-1,
    ),

    "[M-H2O+H]+": Adduct(
        "[M-H2O+H]+",
        PROTON - H2O,
    ),
    "[M-2H2O+H]+": Adduct(
        "[M-2H2O+H]+",
        PROTON - 2 * H2O,
    ),
    "[M-H2O-H]-": Adduct(
        "[M-H2O-H]-",
        -H2O - PROTON,
        charge=-1,
    ),

    # Multimers
    "[2M+H]+": Adduct(
        "[2M+H]+",
        PROTON,
        molecule_count=2,
    ),
    "[2M+Na]+": Adduct(
        "[2M+Na]+",
        NA,
        molecule_count=2,
    ),
    "[2M-H]-": Adduct(
        "[2M-H]-",
        -PROTON,
        molecule_count=2,
        charge=-1,
    ),

    # Multiply charged
    "[M+2H]2+": Adduct(
        "[M+2H]2+",
        2 * PROTON,
        charge=2,
    ),
    "[M+3H]3+": Adduct(
        "[M+3H]3+",
        3 * PROTON,
        charge=3,
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