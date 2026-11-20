from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from django.conf import settings

from management.models.org import Org

if TYPE_CHECKING:
    from management.models import Team, UserProfile


@dataclass(frozen=True)
class ExclusionFields:
    names: str
    merges: str
    default: str


class Exclusion(Enum):
    MODELS = ExclusionFields(
        names="excluded_models",
        merges="merge_exclusion_lists",
        default="AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST",
    )
    MCP_SERVERS = ExclusionFields(
        names="excluded_mcp_servers",
        merges="merge_mcp_server_exclusion_lists",
        default="AQUEDUCT_DEFAULT_MCP_SERVER_EXCLUSION_LIST",
    )


def excluded(holder: "Org | Team | UserProfile", kind: Exclusion) -> frozenset[str]:
    """Walk holder, its org, then the settings default, while each level merges upward."""
    fields = kind.value
    chain = (holder,) if isinstance(holder, Org) else (holder, holder.org)
    found: set[str] = set()
    for level in chain:
        found.update(getattr(level, fields.names))
        if not getattr(level, fields.merges):
            return frozenset(found)
    return frozenset(found.union(getattr(settings, fields.default, ())))
