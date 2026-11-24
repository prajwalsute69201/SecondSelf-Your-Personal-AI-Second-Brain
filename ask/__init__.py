"""SecondSelf — Question Answering & Retrieval-Augmented Generation Package (Task 4.1).

Exposes importable ask() interface, data models, hybrid retriever, and answer synthesizer.
"""

from ask.cli import ask
from ask.context import ContextBuilder
from ask.models import Answer, Citation, ContextBundle, RetrievedItem
from ask.retriever import HybridRetriever
from ask.synthesizer import Synthesizer

__all__ = [
    "Answer",
    "Citation",
    "ContextBuilder",
    "ContextBundle",
    "HybridRetriever",
    "RetrievedItem",
    "Synthesizer",
    "ask",
]
