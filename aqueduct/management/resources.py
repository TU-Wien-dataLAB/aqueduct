from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from django.conf import settings
from django.core.paginator import Page, Paginator
from django.db.models import Count, Q, QuerySet, Sum
from django.db.models.functions import Coalesce
from django.template.defaultfilters import filesizeformat
from django.urls import reverse
from django.utils import timezone
from django.utils.functional import Promise
from django.utils.http import urlencode
from django.utils.translation import gettext, ngettext, ngettext_lazy
from django.utils.translation import gettext_lazy as _

from management.access import EveryOrg, Extent, KeyAct, Memberships, TeamAct, Viewer, WholeOrg
from management.formatting import grouped
from management.icons import Icon
from management.keys import Expiry
from management.models import (
    Batch,
    BatchStatus,
    FileObject,
    Token,
    VectorStore,
    VectorStoreFile,
    VectorStoreFileBatch,
    VectorStoreFileStatus,
    VectorStoreStatus,
)
from management.ownership import of_teams

DAY_SECONDS = 86_400
PAGE_SIZE = 50
NEWEST_FIRST = ("-created_at", "-id")


API_DOCS = "https://tu-wien-datalab.github.io/aqueduct/api/"


def empty_text(shown: Extent, every_org: str, org: str, yours: str) -> str:
    match shown:
        case EveryOrg():
            return every_org
        case WholeOrg(found):
            return org % {"org": found.name}
        case Memberships():
            return yours


def moment(timestamp: int | None) -> datetime | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, tz=UTC)


def owner_of(token: Token, viewer: Viewer) -> str:
    if token.service_account:
        return token.service_account.team.name
    return gettext("You") if viewer.may_key(KeyAct.OWN, token) else token.user.email


@dataclass(frozen=True)
class Status:
    label: str | Promise
    tone: str
    icon: Icon

    @classmethod
    def of(cls, raw: str, known: Mapping[str, "Status"]) -> "Status":
        return known.get(raw) or cls(raw.replace("_", " ").capitalize(), "st-off", Icon.OFF)


def _working(label: Promise) -> Status:
    return Status(label, "text-primary", Icon.PENDING)


_DONE = Status(_("Completed"), "st-ok", Icon.OK)
_FAILED = Status(_("Failed"), "st-err", Icon.FAILED)
_CANCELLED = Status(_("Cancelled"), "st-off", Icon.OFF)
_EXPIRED = Status(_("Expired"), "st-off", Icon.OFF)

BATCH_STATUSES: Mapping[str, Status] = {
    BatchStatus.VALIDATING: _working(_("Validating")),
    BatchStatus.IN_PROGRESS: _working(_("In progress")),
    BatchStatus.FINALIZING: _working(_("Finalizing")),
    BatchStatus.CANCELLING: _working(_("Cancelling")),
    BatchStatus.COMPLETED: _DONE,
    BatchStatus.FAILED: _FAILED,
    BatchStatus.EXPIRED: _EXPIRED,
    BatchStatus.CANCELLED: _CANCELLED,
}
CANCELLABLE = frozenset({BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.FINALIZING})
_READY = Status(_("Ready"), "st-ok", Icon.OK)
STORE_STATUSES: Mapping[str, Status] = {
    VectorStoreStatus.COMPLETED: _READY,
    VectorStoreStatus.IN_PROGRESS: _working(_("Indexing")),
    VectorStoreStatus.EXPIRED: _EXPIRED,
}
STORE_FILE_STATUSES: Mapping[str, Status] = {
    VectorStoreFileStatus.COMPLETED: _READY,
    VectorStoreFileStatus.IN_PROGRESS: _working(_("Indexing")),
    VectorStoreFileStatus.FAILED: _FAILED,
    VectorStoreFileStatus.CANCELLED: _CANCELLED,
}
PURPOSES: Mapping[str, str | Promise] = {
    "batch": _("Batch input"),
    "batch_output": _("Batch output"),
    "assistants": _("Assistants"),
    "assistants_output": _("Assistants output"),
    "user_data": _("User data"),
    "fine-tune": _("Fine-tune"),
    "vision": _("Vision"),
    "evals": _("Evals"),
}


def paged(rows: QuerySet, page: str | int | None) -> tuple[Page, QuerySet]:
    pages = Paginator(
        rows.order_by(*NEWEST_FIRST).values_list("id", flat=True), PAGE_SIZE
    ).get_page(page)
    return pages, rows.model._default_manager.filter(id__in=list(pages)).order_by(*NEWEST_FIRST)


def list_url(name: str, page: str | None) -> str:
    return f"{reverse(name)}?{urlencode({'page': page})}" if page else reverse(name)


def page_holding(rows: QuerySet, row_id: str) -> int | None:
    row = rows.filter(id=row_id).values("created_at").first()
    if row is None:
        return None
    newer = rows.filter(
        Q(created_at__gt=row["created_at"]) | Q(created_at=row["created_at"], id__gt=row_id)
    ).count()
    return newer // PAGE_SIZE + 1


BATCH_SLOTS = ngettext_lazy(
    "%(used)s of your %(limit)s batch slot in use",
    "%(used)s of your %(limit)s batch slots in use",
    "limit",
)
VECTOR_STORE_SLOTS = ngettext_lazy(
    "%(used)s of your %(limit)s vector store slot in use",
    "%(used)s of your %(limit)s vector store slots in use",
    "limit",
)


def quotas(you: Promise, viewer: Viewer, rows: QuerySet, limits: tuple[int, int]) -> str:
    user_limit, team_limit = limits
    teams = viewer.teams(TeamAct.MINE)
    per_team = dict(
        rows.filter(of_teams(teams, "token__"))
        .order_by()
        .values("token__service_account__team")
        .annotate(n=Count("id"))
        .values_list("token__service_account__team", "n")
    )
    used = (
        # The gateway counts every key a user created, service-account keys included.
        str(
            you
            % {
                "used": rows.filter(token__user_id=viewer.profile.user_id).count(),
                "limit": user_limit,
            }
        ),
        *(
            gettext("%(team)s %(used)s of %(limit)s")
            % {"team": team.name, "used": per_team.get(team.id, 0), "limit": team_limit}
            for team in teams.order_by("name")
        ),
    )
    return " · ".join(used)


@dataclass(frozen=True)
class Progress:
    total: int
    completed: int
    failed: int

    @classmethod
    def of(cls, counts: Mapping[str, int] | None) -> "Progress":
        counts = counts or {}
        return cls(counts.get("total", 0), counts.get("completed", 0), counts.get("failed", 0))

    def _share(self, part: int) -> float:
        return round(100 * part / self.total, 1) if self.total else 0

    @property
    def completed_share(self) -> float:
        return self._share(self.completed)

    @property
    def failed_share(self) -> float:
        return self._share(self.failed)

    @property
    def parts(self) -> tuple[str, ...]:
        done = gettext("%(done)s of %(total)s") % {
            "done": grouped(self.completed + self.failed),
            "total": grouped(self.total),
        }
        if not self.failed:
            return (done,)
        failed = ngettext("%(count)s failed", "%(count)s failed", self.failed)
        return done, failed % {"count": grouped(self.failed)}

    @property
    def label(self) -> str:
        return " · ".join(self.parts)


@dataclass(frozen=True)
class FileRow:
    file: FileObject
    owner: str
    purpose: str | Promise
    created: datetime | None
    expiry: Expiry
    mine: bool


@dataclass(frozen=True)
class FileList:
    rows: tuple[FileRow, ...]
    pages: Page
    size: int
    shown: Extent
    docs = f"{API_DOCS}files/"

    @property
    def summary(self) -> str:
        count = self.pages.paginator.count
        limit = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB
        return " · ".join(
            (
                ngettext("%(count)s file", "%(count)s files", count) % {"count": count},
                filesizeformat(self.size),
                gettext("max %(limit)s MB each") % {"limit": limit},
            )
        )

    @property
    def empty(self) -> str:
        return empty_text(
            self.shown,
            gettext("No files yet."),
            gettext("No files in %(org)s yet."),
            gettext("No files from your keys or your teams' keys yet."),
        )


def file_list(
    viewer: Viewer, now: datetime, page: str | int | None = None, file_id: str | None = None
) -> FileList:
    visible = FileObject.objects.filter(token__in=viewer.keys(KeyAct.SEE))
    size = visible.aggregate(size=Coalesce(Sum("bytes"), 0))["size"]
    pages, files = paged(visible, (file_id and page_holding(visible, file_id)) or page)
    files = files.select_related("token__user", "token__service_account__team")
    return FileList(
        tuple(
            FileRow(
                file=file,
                owner=owner_of(file.token, viewer),
                purpose=PURPOSES.get(file.purpose, file.purpose),
                created=moment(file.created_at),
                expiry=Expiry.of(moment(file.expires_at), now),
                mine=viewer.may_key(KeyAct.OWN, file.token),
            )
            for file in files
        ),
        pages,
        size,
        viewer.shown,
    )


@dataclass(frozen=True)
class BatchRow:
    batch: Batch
    owner: str
    status: Status
    progress: Progress
    created: datetime | None
    deadline: datetime | None
    deadline_today: bool
    mine: bool
    cancellable: bool

    @property
    def results(self) -> tuple[tuple[str, FileObject], ...]:
        named = (
            (gettext("Output"), self.batch.output_file),
            (gettext("Errors"), self.batch.error_file),
        )
        return tuple((label, file) for label, file in named if file)

    @property
    def note(self) -> str:
        if self.batch.status in CANCELLABLE:
            return gettext("Output pending")
        return gettext("No output file")


@dataclass(frozen=True)
class BatchList:
    rows: tuple[BatchRow, ...]
    pages: Page
    summary: str
    shown: Extent
    docs = f"{API_DOCS}batches/"

    @property
    def empty(self) -> str:
        return empty_text(
            self.shown,
            gettext("No batches yet."),
            gettext("No batches in %(org)s yet."),
            gettext("No batches from your keys or your teams' keys yet."),
        )


ACTIVE_BATCH = Q(
    status__in=[BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.CANCELLING]
)


def batch_row(batch: Batch, viewer: Viewer, now: datetime) -> BatchRow:
    running = batch.status in CANCELLABLE
    mine = viewer.may_key(KeyAct.OWN, batch.token)
    deadline = moment(batch.expires_at) if running else None
    return BatchRow(
        batch=batch,
        owner=owner_of(batch.token, viewer),
        status=Status.of(batch.status, BATCH_STATUSES),
        progress=Progress.of(batch.request_counts),
        created=moment(batch.created_at),
        deadline=deadline,
        deadline_today=deadline is not None
        and timezone.localdate(deadline) == timezone.localdate(now),
        mine=mine,
        cancellable=running and mine,
    )


def batch_list(viewer: Viewer, now: datetime, page: str | None = None) -> BatchList:
    pages, batches = paged(Batch.objects.filter(token__in=viewer.keys(KeyAct.SEE)), page)
    batches = batches.select_related(
        "token__user", "token__service_account__team", "input_file", "output_file", "error_file"
    )
    limits = (settings.MAX_USER_BATCHES, settings.MAX_TEAM_BATCHES)
    return BatchList(
        tuple(batch_row(batch, viewer, now) for batch in batches),
        pages,
        quotas(BATCH_SLOTS, viewer, Batch.objects.filter(ACTIVE_BATCH), limits),
        viewer.shown,
    )


def store_expiry(store: VectorStore, now: datetime) -> Expiry:
    policy = store.expires_after or {}
    days = policy.get("days")
    anchor = store.last_active_at or store.created_at
    if not isinstance(days, int):
        return Expiry.of(None, now)
    return Expiry.of(moment(anchor + days * DAY_SECONDS), now)


@dataclass(frozen=True)
class StoreRow:
    store: VectorStore
    owner: str
    status: Status
    files: int
    size: int
    expiry: Expiry
    syncable: bool


@dataclass(frozen=True)
class StoreList:
    rows: tuple[StoreRow, ...]
    pages: Page
    summary: str
    shown: Extent
    docs = f"{API_DOCS}vector-stores/"

    @property
    def empty(self) -> str:
        return empty_text(
            self.shown,
            gettext("No vector stores yet."),
            gettext("No vector stores in %(org)s yet."),
            gettext("No vector stores from your keys or your teams' keys yet."),
        )


def store_rows(
    stores: QuerySet[VectorStore], viewer: Viewer, now: datetime
) -> tuple[StoreRow, ...]:
    stores = stores.select_related("token__user", "token__service_account__team").annotate(
        file_count=Count("files"), size=Coalesce(Sum("files__file_obj__bytes"), 0)
    )
    synced = set(stores.filter(token__in=viewer.keys(KeyAct.SYNC)).values_list("id", flat=True))
    return tuple(
        StoreRow(
            store=store,
            owner=owner_of(store.token, viewer),
            status=Status.of(store.status, STORE_STATUSES),
            files=store.file_count,
            size=store.size,
            expiry=store_expiry(store, now),
            syncable=store.id in synced,
        )
        for store in stores
    )


def store_list(viewer: Viewer, now: datetime, page: str | None = None) -> StoreList:
    pages, stores = paged(VectorStore.objects.filter(token__in=viewer.keys(KeyAct.SEE)), page)
    limits = (settings.MAX_USER_VECTOR_STORES, settings.MAX_TEAM_VECTOR_STORES)
    return StoreList(
        store_rows(stores, viewer, now),
        pages,
        quotas(VECTOR_STORE_SLOTS, viewer, VectorStore.objects.all(), limits),
        viewer.shown,
    )


@dataclass(frozen=True)
class StoreFileRow:
    entry: VectorStoreFile
    status: Status
    created: datetime | None
    listed: bool


@dataclass(frozen=True)
class StoreBatchRow:
    batch: VectorStoreFileBatch
    status: Status
    progress: Progress
    in_progress: int
    created: datetime | None


@dataclass(frozen=True)
class StoreDetail:
    row: StoreRow
    files: tuple[StoreFileRow, ...]
    pages: Page
    batches: tuple[StoreBatchRow, ...]
    created: datetime | None
    last_active: datetime | None


def store_detail(
    store_id: str, viewer: Viewer, now: datetime, page: str | None = None
) -> StoreDetail | None:
    listed_keys = viewer.keys(KeyAct.SEE)
    found = store_rows(VectorStore.objects.filter(token__in=listed_keys, id=store_id), viewer, now)
    if not found:
        return None
    row = found[0]
    store = row.store
    pages, entries_on_page = paged(store.files.all(), page)
    entries = tuple(entries_on_page.select_related("file_obj"))
    listed = set(
        FileObject.objects.filter(
            token__in=listed_keys, id__in=[entry.file_obj_id for entry in entries]
        ).values_list("id", flat=True)
    )
    return StoreDetail(
        row=row,
        files=tuple(
            StoreFileRow(
                entry,
                Status.of(entry.status, STORE_FILE_STATUSES),
                moment(entry.created_at),
                entry.file_obj_id in listed,
            )
            for entry in entries
        ),
        pages=pages,
        batches=tuple(
            StoreBatchRow(
                batch,
                Status.of(batch.status, BATCH_STATUSES),
                Progress.of(batch.file_counts),
                (batch.file_counts or {}).get("in_progress", 0),
                moment(batch.created_at),
            )
            for batch in store.file_batches.all()
        ),
        created=moment(store.created_at),
        last_active=moment(store.last_active_at),
    )
