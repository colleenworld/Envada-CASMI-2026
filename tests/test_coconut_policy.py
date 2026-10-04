from casmi26.chemistry.coconut_policy import (
    CoconutDisposition,
    approved_collections,
    classify_collections,
    parse_collections,
)


def test_parse_collections() -> None:
    assert parse_collections(
        "FooDB|Wikidata Natural Products|NPASS"
    ) == {
        "FooDB",
        "Wikidata Natural Products",
        "NPASS",
    }


def test_parse_empty_collections() -> None:
    assert parse_collections(None) == set()
    assert parse_collections(float("nan")) == set()
    assert parse_collections("") == set()


def test_approved_source_wins() -> None:
    collections = parse_collections(
        "FooDB|Wikidata Natural Products|Supernatural3"
    )

    assert (
        classify_collections(collections)
        == CoconutDisposition.APPROVED
    )

    assert approved_collections(collections) == {
        "Wikidata Natural Products"
    }


def test_excluded_only() -> None:
    collections = parse_collections(
        "FooDB|Supernatural3"
    )

    assert (
        classify_collections(collections)
        == CoconutDisposition.EXCLUDED
    )


def test_review_required_only() -> None:
    collections = parse_collections(
        "Super Natural II|ZINC NP"
    )

    assert (
        classify_collections(collections)
        == CoconutDisposition.REVIEW_REQUIRED
    )


def test_unknown_source() -> None:
    collections = parse_collections(
        "Some Future COCONUT Database"
    )

    assert (
        classify_collections(collections)
        == CoconutDisposition.NEEDS_REVIEW
    )


def test_unknown_takes_precedence_over_excluded() -> None:
    collections = parse_collections(
        "FooDB|Some Future COCONUT Database"
    )

    assert (
        classify_collections(collections)
        == CoconutDisposition.NEEDS_REVIEW
    )


def test_no_provenance() -> None:
    assert (
        classify_collections(set())
        == CoconutDisposition.NO_PROVENANCE
    )