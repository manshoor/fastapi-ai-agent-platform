"""Knowledge graph rows from model output: only what each chunk itself states is kept.

    model output (nodes, edges per chunk label)
      node: type in the ontology, acceptable name, name in its chunk (word edges)
      edge: allowed triple, quote of 3+ words in its chunk, both names in the quote,
            two distinct keys
      -> rows keyed by node_key = the name's normalized words, without its type, so one
         entity typed differently in two documents is still one node; capped and
         deduplicated per chunk

A dropped item is counted, never stored. The relation, its direction and the node types
are the model's reading of the quote; the server checks only that the triple is allowed
and the quote names both ends. Retrieval follows edges as undirected links.

Rows belong to one chunk of one version, so the chunk's delete cascades to them.
Entities are merged across documents only at query time, by node_key, inside one
owner's or one building's documents. Pure functions.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.ai.contracts import GraphExtraction
from app.rag.chunking import Chunk
from app.rag.ontology import NodeType, Relation, acceptable_name, allowed
from app.rag.verification import appears_in, check_quote, normalize

MAX_NODE_KEY_CHARACTERS = 160
# Longest node name, in words, a question can name (question phrases are this long at most).
MAX_MATCH_WORDS = 6
_NON_WORD = re.compile(r"[^\w]+")


@dataclass(frozen=True)
class NodeRow:
    ordinal: int
    node_key: str
    node_type: NodeType
    name: str
    match_name: str


@dataclass(frozen=True)
class EdgeRow:
    ordinal: int
    source_key: str
    relation: Relation
    target_key: str
    quote: str


@dataclass
class GraphRows:
    nodes: list[NodeRow] = field(default_factory=list)
    edges: list[EdgeRow] = field(default_factory=list)
    # Items the server refused: ungrounded, outside the ontology, or over a cap.
    dropped: int = 0


def node_key(name: str) -> str:
    """The entity's identity: its normalized words, without the type. A model may type
    "cellar" as a location in one document and a facility in another; both are one key."""
    return match_name(name)


def match_name(name: str) -> str:
    """Normalized words joined by single spaces: "Level-2 Gym" -> "level 2 gym"."""
    return _NON_WORD.sub(" ", normalize(name)).strip()[:MAX_NODE_KEY_CHARACTERS]


def question_phrases(question: str) -> list[str]:
    """Every run of 1..MAX_MATCH_WORDS words of the question, in match_name form.

    A node whose match_name is one of these is named by the question at word edges.
    """
    words = _NON_WORD.sub(" ", normalize(question)).split()
    phrases = {
        " ".join(words[start : start + size])
        for size in range(1, MAX_MATCH_WORDS + 1)
        for start in range(len(words) - size + 1)
    }
    return sorted(phrases)


class _Chunk:
    """The kept rows of one chunk, with its caps."""

    def __init__(self, chunk: Chunk, limit: int):
        self.chunk = chunk
        self.limit = limit
        self.nodes: dict[str, NodeRow] = {}
        self.edges: dict[tuple[str, Relation, str], EdgeRow] = {}

    def node(self, node_type: NodeType, name: str) -> str | None:
        """The node's key if it is grounded and fits; adds it once."""
        if not acceptable_name(name) or not appears_in(name, self.chunk.content):
            return None
        key = node_key(name)
        if key in self.nodes:
            return key
        if len(self.nodes) >= self.limit:
            return None
        self.nodes[key] = NodeRow(self.chunk.ordinal, key, node_type, name, match_name(name))
        return key


def ground(extraction: GraphExtraction, chunks: Sequence[Chunk], limit: int) -> GraphRows:
    """Rows for `chunks` (C1 is chunks[0]); at most `limit` nodes and edges per chunk."""
    kept = [_Chunk(chunk, limit) for chunk in chunks]
    rows = GraphRows()

    def target(label: str) -> _Chunk | None:
        index = int(label[1:]) - 1
        return kept[index] if index < len(kept) else None

    # Nodes first, so an edge's endpoints are usually already counted.
    for item in extraction.nodes:
        chunk = target(item.chunk)
        if chunk is None or chunk.node(item.type, item.name) is None:
            rows.dropped += 1

    for item in extraction.edges:
        chunk = target(item.chunk)
        if (
            chunk is None
            or len(chunk.edges) >= limit
            or not _edge_grounded(item, chunk.chunk.content)
        ):
            rows.dropped += 1
            continue
        source = chunk.node(item.source_type, item.source)
        destination = chunk.node(item.target_type, item.target)
        identity = (source, item.relation, destination)
        # Distinct names can share a key once normalized and cut to the column width.
        if source is None or destination is None or source == destination:
            rows.dropped += 1
            continue
        if identity in chunk.edges:
            continue
        chunk.edges[identity] = EdgeRow(
            chunk.chunk.ordinal, source, item.relation, destination, item.quote
        )

    for chunk in kept:
        rows.nodes.extend(chunk.nodes.values())
        rows.edges.extend(chunk.edges.values())
    return rows


def _edge_grounded(item, text: str) -> bool:
    if not allowed(item.source_type, item.relation, item.target_type):
        return False
    if normalize(item.source) == normalize(item.target):
        return False
    # The quote is a real span of the chunk, and it names both ends of the edge.
    if check_quote("", item.quote, text) is not None:
        return False
    return appears_in(item.source, item.quote) and appears_in(item.target, item.quote)
