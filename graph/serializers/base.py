"""SecondSelf — Abstract Graph Serializer Protocol."""

from __future__ import annotations

import abc
from graph.models import Graph


class GraphSerializer(abc.ABC):
    """Abstract interface for serializing canonical Graph instances."""

    @property
    @abc.abstractmethod
    def extension(self) -> str:
        """File extension associated with this serializer (e.g. '.json', '.cypher')."""
        pass

    @abc.abstractmethod
    def serialize(self, graph: Graph) -> str:
        """Transforms in-memory Graph instance into serialized text payload."""
        pass
