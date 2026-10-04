from __future__ import annotations

from enum import StrEnum


class CoconutDisposition(StrEnum):
    APPROVED = "approved"
    REVIEW_REQUIRED = "review_required"
    EXCLUDED = "excluded"
    NEEDS_REVIEW = "needs_review"
    NO_PROVENANCE = "no_provenance"


APPROVED = {
    "Wikidata Natural Products",
    "ChEBI NPs",
    "GNPS (Global Natural Products Social Molecular Networking)",
}

EXCLUDED = {
    "FooDB",
    "CMNPD",
    "Supernatural3",
    "NPASS",
    "NPAtlas",
    "DrugBankNP",
}

REVIEW_REQUIRED = {
    "Super Natural II",
    "ZINC NP",
    "InterBioScreen Ltd",
    "TCMDB-Taiwan (Traditional Chinese Medicine database)",
    "NPEdia",
    "UNPD (Universal Natural Products Database)",
    "CMAUP (cCollective molecular activities of useful plants)",
    "KNApSaCK",
    "PubChem NPs",
    "ChEMBL NPs",
}


def parse_collections(value: object) -> frozenset[str]:
    """Parse COCONUT's pipe-separated collections field."""
    if value is None:
        return frozenset()

    # Avoid importing pandas just for pd.isna().
    if isinstance(value, float) and value != value:
        return frozenset()

    return frozenset(
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    )


def classify_collections(
    collections: set[str] | frozenset[str],
) -> CoconutDisposition:
    """
    Classify a COCONUT record according to its provenance.

    An approved provenance path takes precedence. Thus a record
    appearing in both an approved and a restrictive collection is
    usable through the approved provenance path.
    """
    if collections & APPROVED:
        return CoconutDisposition.APPROVED

    known = APPROVED | EXCLUDED | REVIEW_REQUIRED
    unknown = collections - known

    if unknown:
        return CoconutDisposition.NEEDS_REVIEW

    if collections & REVIEW_REQUIRED:
        return CoconutDisposition.REVIEW_REQUIRED

    if collections & EXCLUDED:
        return CoconutDisposition.EXCLUDED

    return CoconutDisposition.NO_PROVENANCE


def approved_collections(
    collections: set[str] | frozenset[str],
) -> frozenset[str]:
    """Return the approved provenance paths for a record."""
    return frozenset(collections & APPROVED)


def is_approved(
    collections: set[str] | frozenset[str],
) -> bool:
    return classify_collections(collections) == CoconutDisposition.APPROVED