"""The fixed ontology for building documents: which nodes and relations a graph may hold.

    node    (type, name)            "facility: gym"
    edge    source -relation-> target, with a quote from the chunk that states it

The model may only use these types; a triple outside `_RELATIONS` is dropped. Nodes
are roles and things: a name that looks like a phone number or an email address is
refused, checked on its NFKC form so typographic variants fold first. Person names are
refused only by the prompt; no pattern can tell "Alex Smith" from "Alder Suite".
Pure definitions; no imports from the rest of the app.
"""

import re
import unicodedata
from enum import StrEnum

ONTOLOGY_VERSION = "building-ontology-v1"


class NodeType(StrEnum):
    BUILDING = "building"
    FACILITY = "facility"
    DEVICE = "device"
    SERVICE = "service"
    RULE = "rule"
    SCHEDULE = "schedule"
    CONTACT_ROLE = "contact_role"
    LOCATION = "location"
    DOCUMENT_TOPIC = "document_topic"


class Relation(StrEnum):
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    OPERATED_BY = "operated_by"
    GOVERNED_BY = "governed_by"
    SCHEDULED_ON = "scheduled_on"
    CONTACT_FOR = "contact_for"
    REQUIRES = "requires"
    PROHIBITS = "prohibits"
    ALLOWS = "allows"
    REPLACES = "replaces"


_PLACES = frozenset({NodeType.LOCATION, NodeType.BUILDING})
_THINGS = frozenset({NodeType.FACILITY, NodeType.DEVICE, NodeType.SERVICE, NodeType.LOCATION})
_SUBJECTS = _THINGS | {NodeType.DOCUMENT_TOPIC}

# Relation -> (allowed source types, allowed target types).
_RELATIONS: dict[Relation, tuple[frozenset[NodeType], frozenset[NodeType]]] = {
    Relation.LOCATED_IN: (_THINGS | {NodeType.BUILDING}, _PLACES),
    Relation.PART_OF: (
        frozenset({NodeType.FACILITY, NodeType.DEVICE, NodeType.LOCATION}),
        frozenset({NodeType.FACILITY, NodeType.LOCATION, NodeType.BUILDING}),
    ),
    Relation.OPERATED_BY: (
        frozenset({NodeType.FACILITY, NodeType.DEVICE, NodeType.SERVICE}),
        frozenset({NodeType.SERVICE, NodeType.CONTACT_ROLE}),
    ),
    Relation.GOVERNED_BY: (_THINGS | {NodeType.SCHEDULE}, frozenset({NodeType.RULE})),
    Relation.SCHEDULED_ON: (
        frozenset({NodeType.FACILITY, NodeType.SERVICE, NodeType.RULE}),
        frozenset({NodeType.SCHEDULE}),
    ),
    Relation.CONTACT_FOR: (
        frozenset({NodeType.CONTACT_ROLE, NodeType.SERVICE}),
        _SUBJECTS | {NodeType.RULE},
    ),
    Relation.REQUIRES: (
        _THINGS | {NodeType.RULE},
        frozenset({NodeType.RULE, NodeType.SERVICE, NodeType.DEVICE, NodeType.CONTACT_ROLE}),
    ),
    Relation.PROHIBITS: (frozenset({NodeType.RULE}), _SUBJECTS),
    Relation.ALLOWS: (frozenset({NodeType.RULE}), _SUBJECTS),
    # Same type on both ends: a new device replaces an old device.
    Relation.REPLACES: (
        frozenset({NodeType.DEVICE, NodeType.RULE, NodeType.SCHEDULE, NodeType.SERVICE}),
        frozenset({NodeType.DEVICE, NodeType.RULE, NodeType.SCHEDULE, NodeType.SERVICE}),
    ),
}

MAX_NAME_CHARACTERS = 120
_EMAIL = re.compile(r"\S+@\S+")
# Seven or more digits in a phone-like run: "+44 20 7946 0000", "555-0100 ext".
_PHONE = re.compile(r"\+?\d[\d\s().-]{5,}\d")
_LETTER = re.compile(r"[^\W\d_]")
# Hyphens and dashes that PDF and Word text use inside phone numbers.
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"))


def allowed(source: NodeType, relation: Relation, target: NodeType) -> bool:
    sources, targets = _RELATIONS[relation]
    if relation is Relation.REPLACES and source is not target:
        return False
    return source in sources and target in targets


def acceptable_name(name: str) -> bool:
    """A role or thing: has a letter, fits the column, holds no contact details.

    The name starts and ends on a letter or digit, so its match in a chunk starts and
    ends on word edges, and quote marks never make a second key for one entity.
    """
    if not 1 <= len(name) <= MAX_NAME_CHARACTERS or "\x00" in name:
        return False
    folded = unicodedata.normalize("NFKC", name).translate(_DASHES)
    if not (folded[:1].isalnum() and folded[-1:].isalnum()):
        return False
    if _LETTER.search(folded) is None:
        return False
    return _EMAIL.search(folded) is None and _PHONE.search(folded) is None
