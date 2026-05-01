from __future__ import annotations

from speech_distortion_pipeline.config import PhonologyConfig

from .complexity import ComplexityScorer, HeuristicComplexityScorer
from .features import FeatureExtractor, HeuristicFeatureExtractor
from .g2p import GraphemeToPhoneme, HeuristicEnglishG2P


def build_grapheme_to_phoneme(config: PhonologyConfig) -> GraphemeToPhoneme:
    if config.g2p_backend == "espeak_ng":
        return HeuristicEnglishG2P()
    raise ValueError("Unsupported G2P backend: {0}".format(config.g2p_backend))


def build_feature_extractor(config: PhonologyConfig, g2p: GraphemeToPhoneme) -> FeatureExtractor:
    if config.g2p_backend == "espeak_ng":
        return HeuristicFeatureExtractor(g2p=g2p)
    raise ValueError("Unsupported feature extractor backend for {0}".format(config.g2p_backend))


def build_complexity_scorer(config: PhonologyConfig) -> ComplexityScorer:
    if config.compute_complexity:
        return HeuristicComplexityScorer()
    raise ValueError("Complexity scoring is disabled in config.")
