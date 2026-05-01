import numpy as np

import speech_distortion_pipeline.phonology.acn_neighbors as acn_neighbors


class _FakeTensor:
    def __init__(self, array: list[list[float]]) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeEmbedder:
    def __init__(self) -> None:
        self.model = type(
            "_FakeModel",
            (),
            {
                "subword_to_idx": {
                    "S": 0,
                    "SH": 1,
                    "Z": 2,
                    "TH": 3,
                }
            },
        )()

    def get_embedding(self, sequences: list[list[str]]) -> _FakeTensor:
        mapping = {
            ("S",): [0.0, 0.0],
            ("SH",): [0.1, 0.0],
            ("Z",): [0.2, 0.0],
            ("TH",): [0.4, 0.0],
            ("SH", "Z"): [0.3, 0.0],
            ("Z", "SH"): [0.35, 0.0],
            ("SH", "TH"): [0.45, 0.0],
            ("TH", "SH"): [0.5, 0.0],
            ("Z", "TH"): [0.55, 0.0],
            ("TH", "Z"): [0.6, 0.0],
        }
        return _FakeTensor([mapping[tuple(sequence)] for sequence in sequences])


def test_phone_sequence_distance_rankings_include_ranked_two_phone_candidates(monkeypatch) -> None:
    acn_neighbors._rankings_for_inventory.cache_clear()
    acn_neighbors._sequence_distance_entries_for_inventory.cache_clear()
    monkeypatch.setattr(acn_neighbors, "_load_text_embedder", lambda _model_dir: _FakeEmbedder())

    rankings = acn_neighbors.phone_sequence_distance_rankings(
        inventory=("S", "SH", "Z", "TH"),
        model_dir=acn_neighbors.REPO_ROOT,
        top_n_pairs=3,
    )

    assert rankings["S"][:6] == [
        ["SH"],
        ["Z"],
        ["SH", "Z"],
        ["Z", "SH"],
        ["TH"],
        ["SH", "TH"],
    ]


def test_phone_sequence_distance_entries_include_numeric_distances(monkeypatch) -> None:
    acn_neighbors._rankings_for_inventory.cache_clear()
    acn_neighbors._sequence_distance_entries_for_inventory.cache_clear()
    monkeypatch.setattr(acn_neighbors, "_load_text_embedder", lambda _model_dir: _FakeEmbedder())

    entries = acn_neighbors.phone_sequence_distance_entries(
        inventory=("S", "SH", "Z", "TH"),
        model_dir=acn_neighbors.REPO_ROOT,
        top_n_pairs=3,
    )

    assert entries["S"][:3] == [
        {"phones": ["SH"], "distance": 0.1, "rank": 0},
        {"phones": ["Z"], "distance": 0.2, "rank": 1},
        {"phones": ["SH", "Z"], "distance": 0.3, "rank": 2},
    ]
