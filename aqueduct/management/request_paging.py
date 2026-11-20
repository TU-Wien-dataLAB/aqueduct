from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from django.db.models import F, Q, QuerySet
from django.db.models.expressions import CombinedExpression, OrderBy
from django.utils.translation import gettext

from management.ids import MAX_ID, parse_id
from management.models import Request

PAGE_SIZE = 50


_MAX_PAGE = MAX_ID // PAGE_SIZE


class Column(StrEnum):
    TIME = "time"
    MODEL = "model"
    STATUS = "status"
    TOKENS = "tokens"
    RESPONSE = "response"

    @property
    def expression(self) -> F | CombinedExpression:
        if self is Column.TOKENS:
            return F("input_tokens") + F("output_tokens")
        field = {
            Column.TIME: "timestamp",
            Column.MODEL: "model",
            Column.STATUS: "status_code",
            Column.RESPONSE: "response_time_ms",
        }[self]
        return F(field)

    @property
    def starts_descending(self) -> bool:
        return self is not Column.MODEL


@dataclass(frozen=True)
class Sort:
    column: Column
    descending: bool

    @classmethod
    def parse(cls, raw: str) -> "Sort | None":
        try:
            return cls(Column(raw.removeprefix("-")), raw.startswith("-"))
        except ValueError:
            return None

    def __str__(self) -> str:
        return f"{'-' if self.descending else ''}{self.column.value}"

    @property
    def ordering(self) -> tuple[OrderBy, OrderBy]:
        nulls_last = True if self.column in (Column.STATUS, Column.RESPONSE) else None
        if self.descending:
            return self.column.expression.desc(nulls_last=nulls_last), F("id").desc()
        return self.column.expression.asc(nulls_last=nulls_last), F("id").asc()

    @property
    def param(self) -> str | None:
        return None if self == NEWEST else str(self)


NEWEST = Sort(Column.TIME, descending=True)


@dataclass(frozen=True)
class Header:
    sortable: bool
    choice: str | None
    aria_sort: str | None


def sort_headers(sort: Sort, free: bool) -> dict[str, Header]:
    def header(column: Column) -> Header:
        if column is sort.column:
            choice = Sort(column, not sort.descending)
            aria_sort: str | None = "descending" if sort.descending else "ascending"
        else:
            choice, aria_sort = Sort(column, column.starts_descending), None
        return Header(free or column is Column.TIME, choice.param, aria_sort)

    return {column.value: header(column) for column in Column}


@dataclass(frozen=True)
class Cursor:
    timestamp: datetime
    id: int
    rank: int | None = None

    @classmethod
    def parse(cls, raw: str | None) -> "Cursor | None":
        micros, _, rest = (raw or "").partition(",")
        raw_id, _, raw_rank = rest.partition(",")
        pk = parse_id(raw_id)
        if pk is None or parse_id(micros) is None:
            return None
        rank = parse_id(raw_rank)
        try:
            return cls(_EPOCH + timedelta(microseconds=int(micros)), pk, rank)
        except OverflowError:
            return None

    @classmethod
    def at(cls, request: Request, rank: int | None = None) -> "Cursor":
        return cls(request.timestamp, request.id, rank)

    def __str__(self) -> str:
        micros = (self.timestamp - _EPOCH) // timedelta(microseconds=1)
        return f"{micros},{self.id}" + ("" if self.rank is None else f",{self.rank}")

    @property
    def older(self) -> Q:
        return Q(timestamp__lte=self.timestamp) & (
            Q(timestamp__lt=self.timestamp) | Q(id__lt=self.id)
        )

    @property
    def newer(self) -> Q:
        return Q(timestamp__gte=self.timestamp) & (
            Q(timestamp__gt=self.timestamp) | Q(id__gt=self.id)
        )

    def just_before(self, newest_first: bool) -> "Cursor":
        rank = None if self.rank is None else self.rank - 1
        return Cursor(self.timestamp, self.id + (1 if newest_first else -1), rank)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


LAST = "last"


@dataclass(frozen=True)
class Place:
    before: str | None = None
    after: str | None = None
    page: str | None = None


@dataclass(frozen=True)
class Pager:
    here: Place
    first: Place | None
    previous: Place | None
    next: Place | None
    last: Place | None
    start: int | None

    @property
    def links(self) -> tuple[tuple[str, Place | None], ...]:
        return (
            (gettext("First"), self.first),
            (gettext("Previous"), self.previous),
            (gettext("Next"), self.next),
            (gettext("Last"), self.last),
        )


def _nearest(listed: QuerySet[Request], cursor: Cursor | None, *, older: bool) -> list[Request]:
    if older:
        found = listed.order_by("-timestamp", "-id")
        found = found.filter(cursor.older) if cursor else found
    else:
        found = listed.order_by("timestamp", "id")
        found = found.filter(cursor.newer) if cursor else found
    return list(found[: PAGE_SIZE + 1])


def page_by_time(
    listed: QuerySet[Request],
    newest_first: bool,
    params: Mapping[str, str],
    open_id: int | None,
    count: int,
) -> tuple[list[Request], Pager]:
    forward, backward = ("before", "after") if newest_first else ("after", "before")
    before, after = Cursor.parse(params.get("before")), Cursor.parse(params.get("after"))
    cursor, side = (before, "before") if before else (after, "after")
    last = cursor is None and params.get("page") == LAST
    if cursor is not None and side == backward:
        found = _nearest(listed, cursor, older=not newest_first)
        if len(found) <= PAGE_SIZE:
            cursor = None

    start: int | None
    if last:
        found = _nearest(listed, None, older=not newest_first)
        page, has_previous, has_next = found[PAGE_SIZE - 1 :: -1], len(found) > PAGE_SIZE, False
        start = count - len(page) + 1
    elif cursor is not None and side == backward:
        page, has_previous, has_next = found[PAGE_SIZE - 1 :: -1], True, True
        start = None if cursor.rank is None else cursor.rank - PAGE_SIZE
    else:
        found = _nearest(listed, cursor, older=newest_first)
        page, has_previous, has_next = found[:PAGE_SIZE], cursor is not None, len(found) > PAGE_SIZE
        start = 1 if cursor is None else None if cursor.rank is None else cursor.rank + 1
        if cursor is None and open_id is not None and all(r.id != open_id for r in page):
            anchor = listed.prefetch_related(None).filter(pk=open_id).only("timestamp").first()
            if anchor is not None:
                cursor = Cursor.at(anchor).just_before(newest_first)
                found = _nearest(listed, cursor, older=newest_first)
                page, has_previous, has_next = found[:PAGE_SIZE], True, len(found) > PAGE_SIZE
                start = None

    if start is None and page and not has_next:
        start = count - len(page) + 1
    if start is not None and not (start >= 1 and start + len(page) - 1 <= max(count, 1)):
        start = None
    if last:
        here = Place(page=LAST)
    elif page and has_previous:
        here = Place(**{forward: str(Cursor.at(page[0], start).just_before(newest_first))})
    elif cursor is not None:
        here = Place(**{side: str(cursor)})
    else:
        here = Place()
    end = None if start is None else start + len(page) - 1
    return page, Pager(
        here=here,
        first=Place() if has_previous else None,
        previous=(
            (Place(**{backward: str(Cursor.at(page[0], start))}) if page else Place())
            if has_previous
            else None
        ),
        next=Place(**{forward: str(Cursor.at(page[-1], end))}) if has_next else None,
        last=Place(page=LAST) if has_next else None,
        start=start,
    )


def page_by_offset(
    listed: QuerySet[Request], sort: Sort, raw_page: str | None, count: int
) -> tuple[list[Request], Pager]:
    pages = max(1, -(-count // PAGE_SIZE))
    number = pages if raw_page == LAST else min(max(parse_id(raw_page) or 1, 1), _MAX_PAGE)
    offset = (number - 1) * PAGE_SIZE
    found = list(listed.order_by(*sort.ordering)[offset : offset + PAGE_SIZE + 1])
    page, has_next = found[:PAGE_SIZE], len(found) > PAGE_SIZE or number < pages

    def at(n: int) -> Place:
        return Place() if n == 1 else Place(page=str(n))

    return page, Pager(
        here=at(number),
        first=at(1) if number > 1 else None,
        previous=at(number - 1) if number > 1 else None,
        next=at(number + 1) if has_next else None,
        last=Place(page=LAST) if number < pages else None,
        start=offset + 1 if offset + len(page) <= count else None,
    )
