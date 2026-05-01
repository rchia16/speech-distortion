from __future__ import annotations

import os
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from acn_embed.embed.embedder.text_embedder import TextEmbedder
except ImportError:  # pragma: no cover - depends on optional runtime package
    TextEmbedder = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = Path(os.environ.get("ACN_EMBED_MODEL_DIR", REPO_ROOT / "model" / "embedder-64"))
DEFAULT_MODEL_ARCHIVE = REPO_ROOT / "model.tgz"

DEFAULT_PHONE_INVENTORY: tuple[str, ...] = (
    "AE",
    "AH",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)

_STRESSED_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


def _resolve_model_dir(model_dir: Path | None = None) -> Path:
    resolved = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if resolved.is_dir():
        return resolved

    if resolved == DEFAULT_MODEL_DIR and DEFAULT_MODEL_ARCHIVE.is_file():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DEFAULT_MODEL_ARCHIVE, "r:gz") as archive:
            archive.extractall(path=REPO_ROOT, members=_iter_embedder_members(archive))
        if resolved.is_dir():
            return resolved

    raise FileNotFoundError(
        f"ACN model directory '{resolved}' was not found. "
        f"Extract '{DEFAULT_MODEL_ARCHIVE.name}' or set an explicit model directory."
    )


def _iter_embedder_members(archive: tarfile.TarFile):
    for member in archive.getmembers():
        if member.name.startswith("model/embedder-64/"):
            yield member


@lru_cache(maxsize=4)
def _load_text_embedder(model_dir: str) -> TextEmbedder:
    if TextEmbedder is None:
        raise RuntimeError("acn-embed is not installed in the current environment.")
    return TextEmbedder(Path(model_dir), "phone", device="cpu")


def canonicalize_phone_for_acn(phone: str, supported: Sequence[str]) -> str | None:
    if phone in supported:
        return phone
    if phone in _STRESSED_VOWELS:
        stressed = f"{phone}1"
        if stressed in supported:
            return stressed
    return None


def canonicalize_phone_sequence_for_acn(
    phones: Sequence[str],
    supported: Sequence[str],
) -> list[str] | None:
    canonical: list[str] = []
    for phone in phones:
        resolved = canonicalize_phone_for_acn(phone, supported)
        if resolved is None:
            return None
        canonical.append(resolved)
    return canonical


def _embed_sequences(embedder: TextEmbedder, sequences: Sequence[Sequence[str]]) -> np.ndarray:
    batch = [list(sequence) for sequence in sequences]
    try:
        return embedder.get_embedding(batch, log_interval=0).numpy()
    except TypeError:
        # Compatibility with tests or alternate embedder shims that expose a narrower signature.
        return embedder.get_embedding(batch).numpy()


@lru_cache(maxsize=4)
def _rankings_for_inventory(model_dir: str, inventory: tuple[str, ...]) -> dict[str, list[str]]:
    embedder = _load_text_embedder(model_dir)
    supported = tuple(embedder.model.subword_to_idx.keys())

    canonical_map: dict[str, str] = {}
    effective_inventory: list[str] = []
    for phone in inventory:
        canonical = canonicalize_phone_for_acn(phone, supported)
        if canonical is None:
            continue
        canonical_map[phone] = canonical
        effective_inventory.append(phone)

    if not effective_inventory:
        return {}

    embeddings = _embed_sequences(embedder, [[canonical_map[phone]] for phone in effective_inventory])
    rankings: dict[str, list[str]] = {}
    for index, phone in enumerate(effective_inventory):
        distances: list[tuple[float, str]] = []
        for other_index, other_phone in enumerate(effective_inventory):
            if other_phone == phone:
                continue
            distance = float(np.linalg.norm(embeddings[index] - embeddings[other_index]))
            distances.append((distance, other_phone))
        distances.sort(key=lambda item: (item[0], item[1]))
        rankings[phone] = [other_phone for _, other_phone in distances]
    return rankings


def _build_two_phone_candidates(
    neighbors: Sequence[str],
    limit: int,
) -> list[tuple[str, str]]:
    limited = list(neighbors[: max(0, limit)])
    pairs: list[tuple[str, str]] = []
    for first in limited:
        for second in limited:
            if first == second:
                continue
            pairs.append((first, second))
    return pairs


def _candidate_sort_key(candidate: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    return (len(candidate), tuple(candidate))


@lru_cache(maxsize=4)
def _sequence_distance_entries_for_inventory(
    model_dir: str,
    inventory: tuple[str, ...],
    top_n_pairs: int,
) -> dict[str, list[dict[str, object]]]:
    embedder = _load_text_embedder(model_dir)
    supported = tuple(embedder.model.subword_to_idx.keys())
    single_rankings = _rankings_for_inventory(model_dir, inventory)

    canonical_inventory: dict[str, str] = {}
    effective_inventory: list[str] = []
    for phone in inventory:
        canonical = canonicalize_phone_for_acn(phone, supported)
        if canonical is None:
            continue
        canonical_inventory[phone] = canonical
        effective_inventory.append(phone)

    if not effective_inventory:
        return {}

    sequence_rankings: dict[str, list[dict[str, object]]] = {}
    for phone in effective_inventory:
        candidate_sequences: list[list[str]] = [[other] for other in effective_inventory if other != phone]
        candidate_sequences.extend(
            [list(pair) for pair in _build_two_phone_candidates(single_rankings.get(phone, ()), top_n_pairs)]
        )

        deduped_candidates: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidate_sequences:
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped_candidates.append(candidate)

        canonical_candidates: list[list[str]] = []
        canonical_index: list[list[str]] = []
        for candidate in deduped_candidates:
            resolved = canonicalize_phone_sequence_for_acn(candidate, supported)
            if resolved is None:
                continue
            canonical_candidates.append(resolved)
            canonical_index.append(candidate)

        if not canonical_candidates:
            sequence_rankings[phone] = []
            continue

        source_embedding = _embed_sequences(embedder, [[canonical_inventory[phone]]])[0]
        candidate_embeddings = _embed_sequences(embedder, canonical_candidates)

        ranked_candidates: list[tuple[float, list[str]]] = []
        for index, candidate in enumerate(canonical_index):
            distance = float(np.linalg.norm(source_embedding - candidate_embeddings[index]))
            ranked_candidates.append((distance, candidate))
        ranked_candidates.sort(key=lambda item: (item[0], _candidate_sort_key(item[1])))
        sequence_rankings[phone] = [
            {"phones": list(candidate), "distance": float(distance), "rank": index}
            for index, (distance, candidate) in enumerate(ranked_candidates)
        ]
    return sequence_rankings


def phone_distance_rankings(
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
) -> dict[str, list[str]]:
    resolved = _resolve_model_dir(model_dir)
    return _rankings_for_inventory(str(resolved), tuple(inventory))


def ranked_neighbors(
    phone: str,
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
) -> list[str]:
    return list(phone_distance_rankings(inventory=inventory, model_dir=model_dir).get(phone, ()))


def phone_sequence_distance_rankings(
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
    top_n_pairs: int = 8,
) -> dict[str, list[list[str]]]:
    resolved = _resolve_model_dir(model_dir)
    return {
        phone: [list(entry["phones"]) for entry in entries]
        for phone, entries in _sequence_distance_entries_for_inventory(
            str(resolved),
            tuple(inventory),
            top_n_pairs,
        ).items()
    }


def phone_sequence_distance_entries(
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
    top_n_pairs: int = 8,
) -> dict[str, list[dict[str, object]]]:
    resolved = _resolve_model_dir(model_dir)
    return {
        phone: [
            {
                "phones": list(entry["phones"]),
                "distance": float(entry["distance"]),
                "rank": int(entry["rank"]),
            }
            for entry in entries
        ]
        for phone, entries in _sequence_distance_entries_for_inventory(
            str(resolved),
            tuple(inventory),
            top_n_pairs,
        ).items()
    }


def ranked_neighbor_sequences(
    phone: str,
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
    top_n_pairs: int = 8,
) -> list[list[str]]:
    return [
        list(candidate)
        for candidate in phone_sequence_distance_rankings(
            inventory=inventory,
            model_dir=model_dir,
            top_n_pairs=top_n_pairs,
        ).get(phone, ())
    ]


def ranked_neighbor_sequence_entries(
    phone: str,
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    model_dir: Path | None = None,
    top_n_pairs: int = 8,
) -> list[dict[str, object]]:
    return [
        {
            "phones": list(entry["phones"]),
            "distance": float(entry["distance"]),
            "rank": int(entry["rank"]),
        }
        for entry in phone_sequence_distance_entries(
            inventory=inventory,
            model_dir=model_dir,
            top_n_pairs=top_n_pairs,
        ).get(phone, ())
    ]
