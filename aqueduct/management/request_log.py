from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

from django.db.models import Q
from django.utils.translation import gettext

from management.ids import parse_id
from management.outcomes import Outcome
from management.request_counts import FAILED_TAB, STATUS_FILTERS, Tab, tab_counts
from management.request_filters import Filters, describe_key
from management.request_paging import (
    NEWEST,
    Column,
    Header,
    Pager,
    Sort,
    page_by_offset,
    page_by_time,
    sort_headers,
)
from management.request_rows import Row


@dataclass(frozen=True)
class RequestLog:
    rows: tuple[Row, ...]
    tabs: tuple[Tab, ...]
    count: int
    status: str | None
    search: str
    searches_text: bool
    sort: Sort
    headers: Mapping[str, Header]
    pager: Pager
    shows_key: bool
    shows_overhead: bool

    @property
    def text_blocked(self) -> bool:
        return bool(self.search) and not self.searches_text and parse_id(self.search) is None

    @property
    def end(self) -> int | None:
        start = self.pager.start
        return None if start is None else start + len(self.rows) - 1


SEARCH_MAX = 256


def request_search(text: str, *, by_text: bool) -> Q:
    pk = parse_id(text)
    found = Q(pk__in=[]) if pk is None else Q(pk=pk)
    if by_text:
        found |= Q(path__icontains=text) | Q(user_id__icontains=text)
        found |= Q(user_agent__icontains=text)
    return found


BOUNDED_SPAN = timedelta(days=7)


def bounded_scan(filters: Filters) -> bool:
    return (
        filters.org_wide is None
        or filters.model is not None
        or filters.period.end - filters.period.start <= BOUNDED_SPAN
    )


def request_log(filters: Filters, params: Mapping[str, str]) -> RequestLog:
    status = raw if (raw := params.get("status")) in STATUS_FILTERS else None
    search = params.get("q", "").replace("\x00", "").strip()[:SEARCH_MAX]
    free = bounded_scan(filters)
    sort = Sort.parse(params.get("sort", "")) or NEWEST
    if not (free or sort.column is Column.TIME):
        sort = NEWEST
    open_id = parse_id(params.get("request"))

    filtered = filters.period.of(filters.requests)
    searched = request_search(search, by_text=free) if search else None
    if searched is not None:
        filtered = filtered.filter(searched)
    counts = tab_counts(filters, searched)
    tabs = (
        Tab(None, gettext("All"), counts[None]),
        Tab(FAILED_TAB, gettext("Failed"), counts[FAILED_TAB]),
        *(Tab(o.codes, str(o.label), counts[o.codes], o.codes) for o in Outcome),
    )

    if status is not None:
        filtered = filtered.filter(STATUS_FILTERS[status])
    listed = filtered.prefetch_related("token__user__profile", "token__service_account__team")
    if sort.column is Column.TIME:
        page, pager = page_by_time(listed, sort.descending, params, open_id, counts[status])
    else:
        page, pager = page_by_offset(listed, sort, params.get("page"), counts[status])

    rows = tuple(
        Row(request, describe_key(request.token, filters.viewer), request.id == open_id)
        for request in page
    )
    return RequestLog(
        rows=rows,
        tabs=tabs,
        count=counts[status],
        status=status,
        search=search,
        searches_text=free,
        sort=sort,
        headers=sort_headers(sort, free),
        pager=pager,
        shows_key=filters.org_wide is not None or (filters.key is None and len(filters.keys) > 1),
        shows_overhead=filters.can_pick_scope,
    )
