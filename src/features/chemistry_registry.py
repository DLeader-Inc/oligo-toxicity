"""Single source of truth for extensible oligonucleotide chemistry classes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SugarChemistry:
    name: str
    code: str
    aliases: tuple[str, ...]
    modified: bool


@dataclass(frozen=True)
class BaseChemistry:
    name: str
    canonical_base: str
    aliases: tuple[str, ...]
    modified: bool


SUGAR_CHEMISTRIES = (
    SugarChemistry("d", "D", ("d",), False),
    SugarChemistry("r", "R", ("r",), False),
    SugarChemistry("m", "O", ("m", "ome", "2ome", "2'-ome"), True),
    SugarChemistry("lna", "L", ("lna",), True),
    SugarChemistry("moe", "M", ("moe",), True),
    SugarChemistry("cet", "E", ("cet", "c-et", "constrainedethyl"), True),
)

BASE_CHEMISTRIES = (
    BaseChemistry("A", "A", ("A",), False),
    BaseChemistry("C", "C", ("C",), False),
    BaseChemistry("G", "G", ("G",), False),
    BaseChemistry("T", "T", ("T",), False),
    BaseChemistry("U", "T", ("U",), False),
    BaseChemistry("m5C", "C", ("M5C", "5MC", "5MEC"), True),
)

SUGAR_ONEHOT_VOCAB = tuple(item.name for item in SUGAR_CHEMISTRIES) + ("OTHER",)
BASE_ONEHOT_VOCAB = tuple(item.name for item in BASE_CHEMISTRIES) + ("OTHER",)
SUGAR_DESCRIPTOR_CODES = tuple(item.code for item in SUGAR_CHEMISTRIES) + ("OTHER",)
MODIFIED_SUGAR_CODES = frozenset(
    item.code for item in SUGAR_CHEMISTRIES if item.modified
)
MODIFIED_BASE_NAMES = tuple(item.name for item in BASE_CHEMISTRIES if item.modified)


def _normalize(token: object) -> str:
    return (
        str(token)
        .strip()
        .replace("[", "")
        .replace("]", "")
        .replace("_", "")
        .replace("-", "")
        .replace("'", "")
        .lower()
    )


def sugar_chemistry(token: object) -> SugarChemistry | None:
    normalized = _normalize(token)
    for item in SUGAR_CHEMISTRIES:
        if normalized in {_normalize(alias) for alias in item.aliases}:
            return item
    return None


def base_chemistry(token: object) -> BaseChemistry | None:
    normalized = _normalize(token)
    for item in BASE_CHEMISTRIES:
        if normalized in {_normalize(alias) for alias in item.aliases}:
            return item
    return None


def sugar_onehot_category(token: object) -> str:
    item = sugar_chemistry(token)
    return item.name if item is not None else "OTHER"


def sugar_descriptor_code(token: object) -> str:
    item = sugar_chemistry(token)
    # Modification patterns are residue-aligned strings, so unknown chemistry
    # must remain a single-character token.
    return item.code if item is not None else "X"


def base_onehot_category(token: object) -> str:
    item = base_chemistry(token)
    return item.name if item is not None else "OTHER"


def canonical_base(token: object) -> str:
    item = base_chemistry(token)
    return item.canonical_base if item is not None else "N"


def base_modification_name(token: object) -> str | None:
    item = base_chemistry(token)
    return item.name if item is not None and item.modified else None


def registry_manifest() -> dict[str, object]:
    return {
        "sugar_chemistries": [item.__dict__ for item in SUGAR_CHEMISTRIES],
        "base_chemistries": [item.__dict__ for item in BASE_CHEMISTRIES],
        "extension_policy": (
            "Register aliases and semantics once; One-Hot adds one category per "
            "new component class, while descriptor generators iterate the registry."
        ),
    }
