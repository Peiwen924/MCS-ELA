# Copyright (c) ModelScope Contributors. All rights reserved.
from .acc import AccMetrics
from .bioner_f1 import BionerF1Metrics
from .embedding import InfonceMetrics, PairedMetrics
from .nlg import NlgMetrics
from .reranker import RerankerMetrics

# Add your own metric calculation method here, use `--eval_metric xxx` to train
# The metric here will only be called during validation

eval_metrics_map = {
    'acc': AccMetrics,
    'bioner_f1': BionerF1Metrics,
    'nlg': NlgMetrics,
    # embedding
    'infonce': InfonceMetrics,
    'paired': PairedMetrics,
    # reranker
    'reranker': RerankerMetrics,
}
