"""Core data models for the AAD knowledge graph."""

from pydantic import BaseModel, Field


class Association(BaseModel):
    """An association links a node to another concept via a vector and a reason."""

    vector: list[float] = Field(
        default_factory=list,
        description="Embedding vector of the associated concept",
    )
    reason: str = Field(
        default="",
        description="Natural language explanation of the association",
    )


class Node(BaseModel):
    """A node in the AAD knowledge graph.

    Nodes are keyed by `name` (unique). The `vector` field holds the
    embedding of `name + content`. The `associations` list stores related
    concept vectors and their reasons.
    """

    name: str = Field(..., description="Unique node identifier")
    content: str = Field(default="", description="Free-text node content")
    vector: list[float] = Field(
        default_factory=list,
        description="Embedding vector (1024-d for DeepSeek)",
    )
    associations: list[Association] = Field(
        default_factory=list,
        description="Associated concept vectors with reasons",
    )
