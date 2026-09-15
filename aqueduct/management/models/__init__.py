# models/__init__.py
"""Django models for the management app.

The public import surface (`from management.models import <Model>`) is preserved:
every model and helper is re-exported here so callers everywhere in the codebase
keep working unchanged. New models go in the module matching their domain and
must be listed below so Django auto-discovers them (these imports are manual —
Django does not walk submodules automatically).

Historically-referenced names (`generate_file_id`, `generate_batch_id`,
`default_request_counts`) that are imported by historical migrations (0004, 0005)
must keep resolving from this package, hence the explicit re-exports.
"""

from .accounts import Org, ServiceAccount, Team, TeamMembership, Token, UserGroup, UserProfile
from .batches import Batch, BatchStatus, default_request_counts, generate_batch_id
from .files import FileObject, generate_file_id
from .mixins import LimitMixin, LimitSet, MCPServerExclusionMixin, ModelExclusionMixin
from .snippets import Snippet, SnippetType
from .usage import Request, Usage
from .vector_stores import (
    VectorStore,
    VectorStoreFile,
    VectorStoreFileBatch,
    VectorStoreFileBatchStatus,
    VectorStoreFileStatus,
    VectorStoreStatus,
)

__all__ = [
    "Batch",
    "BatchStatus",
    "FileObject",
    "LimitMixin",
    "LimitSet",
    "MCPServerExclusionMixin",
    "ModelExclusionMixin",
    "Org",
    "Request",
    "ServiceAccount",
    "Snippet",
    "SnippetType",
    "Team",
    "TeamMembership",
    "Token",
    "Usage",
    "UserGroup",
    "UserProfile",
    "VectorStore",
    "VectorStoreFile",
    "VectorStoreFileBatch",
    "VectorStoreFileBatchStatus",
    "VectorStoreFileStatus",
    "VectorStoreStatus",
    "default_request_counts",
    "generate_batch_id",
    "generate_file_id",
]
