import datetime as dt
import math
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.contrib.humanize.templatetags.humanize import intcomma
from django.db.models import Count, F, Q, QuerySet, Sum
from django.db.models.functions import TruncDay, TruncHour, TruncMinute
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.http import urlencode
from django.utils.translation import gettext, ngettext
from django.utils.translation import gettext_lazy as _
from django.views.generic import TemplateView

from gateway.config import get_router_config
from management.models import Org, Request, Token
from management.templatetags.ui import compact, percent
from management.views.base import BaseAqueductView
from management.views.requests import SPANS, STATUSES

TOTALS = {
    "requests": Count("id"),
    **{name: Count("id", filter=q) for name, q in STATUSES.items()},
    "input": Sum("input_tokens", default=0),
    "cached": Sum("cached_input_tokens", default=0),
    "output": Sum("output_tokens", default=0),
    "reasoning": Sum("reasoning_tokens", default=0),
}
"""The figures of a group of requests: counts per status and token sums."""
FIGURES = {
    **TOTALS,
    "response_ms": Sum("response_time_ms", default=0),
    "responses": Count("response_time_ms"),
}
"""TOTALS, and the sum and number of response times that an average over groups needs."""
STEPS: dict[str, tuple[timedelta, Any, str, str, str]] = {
    "minute": (
        timedelta(minutes=1),
        TruncMinute,
        "H:i",
        _("Tokens per minute"),
        _("Requests per minute"),
    ),
    "hour": (timedelta(hours=1), TruncHour, "H:00", _("Tokens per hour"), _("Requests per hour")),
    "day": (timedelta(days=1), TruncDay, "j M", _("Tokens per day"), _("Requests per day")),
}
"""Each time bucket's length, its database function, its label format and the chart titles."""
SPAN_STEPS = {"1h": "minute", "1d": "hour", "1w": "day", "1m": "day"}
"""The time bucket of each time frame; the Requests log's SPANS give the time frames' lengths."""
STATUS_ROWS = (
    ("ok", _("OK"), "2xx, 3xx", "ok", "st-ok"),
    ("rate_limited", _("Rate-limited"), "429", "warning", "st-warn"),
    ("client_error", _("Client error"), "4xx", "off", "st-off"),
    ("server_error", _("Server error"), "5xx", "failed", "st-err"),
)
AXIS_LABELS, SPARK_BARS, TOP_ROWS, FAILED_ALERT_PERCENT = 8, 12, 10, 2


def _is_id(value: str) -> bool:
    """ASCII digits only: `str.isdigit()` also accepts characters like "²" that `int()` rejects."""
    return value.isascii() and value.isdigit()


def _with_shares(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add each row's tokens, new input, and whether its failures deserve a red flag."""
    for row in rows:
        row["tokens"] = row["input"] + row["output"]
        row["new_input"] = max(row["input"] - row["cached"], 0)
        row["failing"] = 100 * row["failed"] > FAILED_ALERT_PERCENT * row["requests"]
    return rows


def show_tokens(count: float) -> str:
    return ngettext("%(count)s token", "%(count)s tokens", count) % {"count": compact(count)}


def show_requests(count: float) -> str:
    return ngettext("%(count)s request", "%(count)s requests", count) % {"count": intcomma(count)}


def _bars(
    groups: list[tuple[str, Any]],
    parts_of: Callable[[Any], tuple[tuple[str, str, float], ...]],
    show: Callable[[float], str],
) -> tuple[list[dict[str, Any]], float]:
    """A bar per labelled group of figures, stacked from its (CSS class, name, amount) parts,
    scaled to the tallest bar; and the tallest bar's amount."""
    parts = [parts_of(figures) for _, figures in groups]
    sums = [sum(amount for *_, amount in each) for each in parts]
    top = max(sums, default=0)
    bars = [
        {
            "heading": heading,
            "text": show(total),
            "height": round(100 * total / top) if top else 0,
            "segments": [(css, round(100 * n / top)) for css, _, n in each if n],
            "parts": [(css, name, intcomma(n)) for css, name, n in each] if total else [],
        }
        for (heading, _), each, total in zip(groups, parts, sums, strict=True)
    ]
    return bars, top


def spark(points: list[tuple[str, float]], show: Callable[[float], str]) -> list[dict[str, Any]]:
    """A small bar chart for _spark.html: a bar per (heading, amount), scaled to the largest."""
    return _bars(points, lambda amount: (("i", "", amount),), show)[0]


def usage_rows(reqs: QuerySet[Request], step: str, by: str | None = None) -> list[dict[str, Any]]:
    """The FIGURES of the requests per `step` long time bucket (`period`), model, and the
    request field `by` (`item`), in one pass. The page's totals, charts and tables are sums of
    these rows, not a pass each over the same requests."""
    groups = {"period": STEPS[step][1]("timestamp"), **({"item": F(by)} if by else {})}
    return list(reqs.annotate(**groups).values("model", *groups).annotate(**FIGURES).order_by())


def _grouped(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, list]:
    """The rows per `key(row)`, in the order the keys first appear."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def _sum(rows: list[dict[str, Any]], names: Iterable[str] = TOTALS) -> dict[str, Any]:
    """The figures `names` of all `rows` together."""
    return {name: sum(row[name] for row in rows) for name in names}


def usage_series(
    rows: list[dict[str, Any]], step: str, start: dt.datetime, end: dt.datetime
) -> list[dict[str, Any]]:
    """The figures of each `step` long time bucket from `start` to `end`, summed from
    `usage_rows` of the same step and labelled for the charts. Buckets start at whole minutes,
    hours or days in UTC, as the database cuts them."""
    length, _, label_format, *_ = STEPS[step]
    seconds = int(length.total_seconds())
    found = {
        bucket: _sum(group)
        for bucket, group in _grouped(rows, lambda row: int(row["period"].timestamp())).items()
    }
    first, last = (int(moment.timestamp()) // seconds * seconds for moment in (start, end))
    return _with_shares(
        [
            {
                **found.get(bucket, dict.fromkeys(TOTALS, 0)),
                "label": date_format(
                    timezone.localtime(dt.datetime.fromtimestamp(bucket, dt.UTC)), label_format
                ),
            }
            for bucket in range(first, last + 1, seconds)
        ]
    )


def usage_charts(rows: list[dict[str, Any]], step: str) -> dict[str, Any]:
    """The tokens and the requests chart, a stacked column per time bucket, and their axis."""
    groups = [(row["label"], row) for row in rows]
    tokens, most_tokens = _bars(
        groups,
        lambda row: (
            ("c", gettext("From cache"), row["cached"]),
            ("n", gettext("New input"), row["new_input"]),
            ("o", gettext("Output"), row["output"]),
        ),
        show_tokens,
    )
    requests, most_requests = _bars(
        groups,
        lambda row: (
            ("i", gettext("OK"), row["requests"] - row["failed"] - row["rate_limited"]),
            ("r", gettext("Rate-limited"), row["rate_limited"]),
            ("f", gettext("Failed"), row["failed"]),
        ),
        show_requests,
    )
    every = math.ceil(len(rows) / AXIS_LABELS) or 1
    return {
        "empty": not most_requests,
        "columns": len(rows),
        "axis": [
            {"label": row["label"], "column": index + 1, "span": min(every, len(rows) - index)}
            for index, row in enumerate(rows)
            if index % every == 0
        ],
        "metrics": [
            ("tokens", STEPS[step][3], show_tokens(most_tokens), tokens),
            ("requests", STEPS[step][4], show_requests(most_requests), requests),
        ],
    }


def model_rows(usage: list[dict[str, Any]], link: Callable[[str], str]) -> list[dict[str, Any]]:
    """Each model's figures of `usage_rows`, most tokens first, with its type, share of tokens,
    average response time and log link."""
    # LiteLLM treats a model without a mode as chat.
    modes = {
        m["model_name"]: (m.get("model_info") or {}).get("mode") or "chat"
        for m in get_router_config().get("model_list", [])
    }
    rows = []
    for model, group in _grouped(usage, lambda row: row["model"]).items():
        figures = _sum(group, FIGURES)
        response_ms, responses = figures.pop("response_ms"), figures.pop("responses")
        average = response_ms / responses if responses else None
        rows.append({"model": model, **figures, "avg_ms": average})
    rows = _with_shares(rows)
    tokens = sum(row["tokens"] for row in rows)
    for row in rows:
        row["mode"], row["log_url"] = modes.get(row["model"]), link(row["model"])
        row["share"] = percent(row["tokens"], tokens) if row["tokens"] else "—"
    return sorted(rows, key=lambda row: (-row["tokens"], -row["requests"], row["model"]))


class UsageDashboardView(BaseAqueductView, TemplateView):
    """
    Usage dashboard page: switch between all orgs (global) and per-org token usage.
    """

    template_name = "management/usage.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.profile
        is_global_admin = profile.is_admin() if profile else False

        # Organization selection: 'all' for global, or org id
        orgs = Org.objects.all() if is_global_admin else Org.objects.filter(pk=profile.org_id)

        selected_org = self._get_selected_org(is_global_admin)

        # Base queryset of requests in scope
        # Global admins see all requests; org admins see all for the selected org;
        # standard users see only their own tokens or service accounts of teams they belong to
        if selected_org is None:
            reqs = Request.objects.all()
        # Org-level access: full org view for org admins
        elif profile.is_org_admin(selected_org):
            reqs = Request.objects.filter(
                Q(token__user__profile__org=selected_org)
                | Q(token__service_account__team__org=selected_org)
            )
        else:
            # Standard user: restrict to own tokens or service accounts' token in their teams;
            # also include requests whose `user_id` value matches the user's email, even if
            # the token does not belong to the user, but to another one from the same org.
            user = profile.user
            teams = self.get_teams_for_user()
            reqs = Request.objects.filter(
                Q(token__user=user)
                | Q(token__service_account__team__in=teams)
                | (Q(user_id=user.email) & Q(token__user__profile__org=profile.org))
            )

        # Only include requests for models configured in the router
        router_models = get_router_config().get("model_list", [])
        allowed_models = [m["model_name"] for m in router_models]
        reqs = reqs.filter(model__in=allowed_models)

        # Time frame selection: 1 hour, 1 day, 1 week, 1 month, or custom range
        now = timezone.now()

        # Parse optional custom date range
        custom_start_str = self.request.GET.get("start", "")
        custom_end_str = self.request.GET.get("end", "")
        custom_start, custom_end, custom_error = self._parse_custom_range(
            custom_start_str, custom_end_str, now
        )

        selected_span = self.request.GET.get("span", "")
        if custom_start and custom_end:
            selected_span, start_time, end_time = "custom", custom_start, custom_end
            length = end_time - start_time
            step = (
                "minute"
                if length <= timedelta(hours=2)
                else "hour"
                if length <= timedelta(days=2)
                else "day"
            )
        else:
            selected_span = selected_span if selected_span in SPAN_STEPS else "1d"
            start_time, end_time = now - SPANS[selected_span], now
            step = SPAN_STEPS[selected_span]

        # Filter requests based on selected time span
        reqs_span = reqs.filter(timestamp__gte=start_time)
        if selected_span == "custom":
            reqs_span = reqs_span.filter(timestamp__lt=end_time)

        # The keys to pick from: those of the org the viewer may open that sent requests
        keys = (
            profile.visible_tokens(selected_org)
            .filter(pk__in=reqs_span.values("token_id"))
            .order_by("name")
            if selected_org is not None
            else Token.objects.none()
        )

        # Optional token and model selection: filter to a single token or model if provided
        selected_token = self._get_selected_token(selected_org)
        if selected_token is not None:
            reqs_span = reqs_span.filter(token=selected_token)
        selected_model = self.request.GET.get("model", "")
        if selected_model not in allowed_models:
            selected_model = ""
        if selected_model:
            reqs_span = reqs_span.filter(model=selected_model)

        # One pass over the requests, per time bucket, model, and what the top table lists:
        # orgs for all orgs, else keys, or a picked key's user IDs
        by = (
            "token__user__profile__org_id"
            if selected_org is None
            else "token_id"
            if selected_token is None
            else "user_id"
        )
        rows = usage_rows(reqs_span, step, by)
        totals = _with_shares([_sum(rows)])[0]
        series = usage_series(rows, step, start_time, end_time)
        limit = 100 if self.request.GET.get("rows") == "all" else TOP_ROWS
        top_items = _with_shares(self._top_items(rows, selected_org, selected_token)[: limit + 1])
        openable = set(
            profile.visible_tokens()
            .filter(pk__in=[item["token_id"] for item in top_items if "token_id" in item])
            .values_list("pk", flat=True)
        )

        # Links into the Requests log under the same filters
        log_filters = {
            "span": selected_span if selected_span in SPANS else None,
            "key": selected_token.pk if selected_token else None,
            "model": selected_model,
        }

        def log_url(**params: str) -> str:
            query = {name: v for name, v in {**log_filters, **params}.items() if v}
            return f"{reverse('requests')}?{urlencode(query)}"

        # Retention warning
        retention_days = getattr(settings, "REQUEST_RETENTION_DAYS", 7)
        retention_warning = (now - start_time).days > retention_days

        context.update(
            {
                "title": "Usage",
                "is_global_admin": is_global_admin,
                "orgs": orgs,
                "selected_org": selected_org,
                "selected_token": selected_token,
                "selected_model": selected_model,
                "keys": keys,
                "model_names": allowed_models,
                "spans": [
                    ("1h", _("1 hour")),
                    ("1d", _("24 hours")),
                    ("1w", _("7 days")),
                    ("1m", _("30 days")),
                ],
                "selected_span": selected_span,
                "totals": totals,
                "total_requests": totals["requests"],
                "kpis": self._kpis(series, totals),
                "charts": usage_charts(series, step),
                "statuses": [
                    {
                        "label": label,
                        "codes": codes,
                        "icon": icon,
                        "tone": tone,
                        "count": totals[value],
                        "share": percent(totals[value], totals["requests"]),
                        "url": log_url(status=value),
                    }
                    for value, label, codes, icon, tone in STATUS_ROWS
                ],
                "log_url": log_url(),
                "per_model": model_rows(rows, lambda model: log_url(model=model)),
                "top_items": top_items[:limit],
                "more_items": len(top_items) > limit,
                "items_busiest": max((item["tokens"] for item in top_items), default=0),
                "openable": openable,
                "retention_warning": retention_warning,
                "retention_days": int(retention_days),
                "custom_start": custom_start_str,
                "custom_end": custom_end_str,
                "custom_error": custom_error,
                "now_utc_iso": now.strftime("%Y-%m-%dT%H:%M"),
            }
        )
        return context

    def _get_selected_org(self, is_global_admin: bool) -> Org | None:
        """Optional org selection based on the query param and user permissions"""
        sel = self.request.GET.get("org")
        if sel == "all" and is_global_admin:
            selected_org = None
        else:
            try:
                selected_org = (
                    Org.objects.get(pk=int(sel))
                    if sel and _is_id(sel) and is_global_admin
                    else self.org
                )
            except Org.DoesNotExist:
                selected_org = self.org
        return selected_org

    @staticmethod
    def _validate_custom_range(
        start: dt.datetime | None, end: dt.datetime | None, now: dt.datetime
    ) -> str | None:
        """Return the first validation error for a custom range, or ``None``."""
        if not (start and end):
            return None
        if end > now:
            return "End time is in the future."
        if start >= end:
            return "Start must be before end."
        if (end - start) < timedelta(minutes=1):
            return "Range must be at least 1 minute."
        return None

    def _parse_custom_range(
        self, start_str: str, end_str: str, now: dt.datetime
    ) -> tuple[dt.datetime | None, dt.datetime | None, str | None]:
        """Parse and validate the optional custom start/end query params.

        Returns ``(start, end, error)``:
            - ``start`` / ``end`` are timezone-aware datetimes in UTC, or ``None``
              if missing or invalid.
            - ``error`` is a human-readable message describing the first
              validation problem found, or ``None`` when the range is valid.
        """
        if not start_str and not end_str:
            return None, None, None
        try:
            start = dt.datetime.fromisoformat(start_str) if start_str else None
            end = dt.datetime.fromisoformat(end_str) if end_str else None
        except (ValueError, TypeError):
            return None, None, "Invalid date format."
        if start is not None and timezone.is_naive(start):
            start = timezone.make_aware(start, dt.UTC)
        if end is not None and timezone.is_naive(end):
            end = timezone.make_aware(end, dt.UTC)
        return start, end, self._validate_custom_range(start, end, now)

    def _get_selected_token(self, selected_org: Org | None) -> Token | None:
        """Optional token selection based on the query param and user permissions"""
        token_param = self.request.GET.get("token")
        if not (token_param and _is_id(token_param)):
            return None
        # A key the viewer may not see counts as not selected.
        return self.profile.visible_tokens(selected_org).filter(pk=int(token_param)).first()

    @staticmethod
    def _kpis(rows: list[dict[str, Any]], totals: dict[str, Any]) -> list[dict[str, Any]]:
        """The four stat cards, each with a bar chart of at most `SPARK_BARS` bars."""
        size = math.ceil(len(rows) / SPARK_BARS) or 1
        slots = []
        for start in range(0, len(rows), size):
            group = rows[start : start + size]
            first, last = group[0]["label"], group[-1]["label"]
            between = gettext("%(first)s to %(last)s") % {"first": first, "last": last}
            figures = {name: sum(row[name] for row in group) for name in TOTALS}
            figures["failed_share"] = (
                100 * figures["failed"] / figures["requests"] if figures["requests"] else 0
            )
            slots.append((first if first == last else between, figures))

        def card(
            label: str, value: str, note: str, field: str, show: Callable[[float], str]
        ) -> dict[str, Any]:
            bars = spark([(heading, figures[field]) for heading, figures in slots], show)
            return {"label": label, "value": value, "note": note, "spark": bars}

        share = percent(totals["cached"], totals["input"])
        return [
            card(gettext("Requests"), intcomma(totals["requests"]), "", "requests", show_requests),
            card(
                gettext("Input tokens"),
                compact(totals["input"]),
                gettext("%(share)s from cache") % {"share": share} if totals["input"] else "",
                "input",
                show_tokens,
            ),
            card(
                gettext("Output tokens"),
                compact(totals["output"]),
                gettext("%(count)s reasoning") % {"count": compact(totals["reasoning"])}
                if totals["reasoning"]
                else "",
                "output",
                show_tokens,
            ),
            card(
                gettext("Failed"),
                percent(totals["failed"], totals["requests"]),
                ngettext("%(count)s rate-limited", "%(count)s rate-limited", totals["rate_limited"])
                % {"count": intcomma(totals["rate_limited"])},
                "failed_share",
                lambda v: gettext("%(share)s failed") % {"share": f"{round(v)}%"},
            ),
        ]

    @staticmethod
    def _top_items(
        rows: list[dict[str, Any]], selected_org: Org | None, selected_token: Token | None
    ) -> list[dict[str, Any]]:
        """The figures of each `item` of `usage_rows`, most tokens first, with its names."""
        items = _grouped(rows, lambda row: row["item"])
        if selected_org is None:
            # Top orgs, for global admins. A service account key counts for its user's org.
            names = dict(Org.objects.filter(pk__in=items).values_list("pk", "name"))
            labels = {org_id: {"name": names.get(org_id), "org_id": org_id} for org_id in items}
        elif selected_token is None:
            # Top keys of the selected org
            labels = {
                key["token_id"]: key
                for key in Token.objects.filter(pk__in=items).values(
                    "name",
                    token_id=F("id"),
                    owner_id=F("user_id"),
                    user_email=F("user__email"),
                    service_account_name=F("service_account__name"),
                    team_name=F("service_account__team__name"),
                )
            }
        else:
            # Top user IDs of the selected key
            labels = {
                user_id: {"name": selected_token.name, "user_id": user_id} for user_id in items
            }
        top = []
        for item, group in items.items():
            figures = _sum(group)
            top.append(
                # A key deleted since the pass has no labels.
                {
                    **labels.get(item, {}),
                    **figures,
                    "total_sum": figures["input"] + figures["output"],
                }
            )
        return sorted(top, key=lambda row: (-row["total_sum"], -row["requests"]))
