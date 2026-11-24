import datetime as dt
import math
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.contrib.humanize.templatetags.humanize import intcomma
from django.db.models import Avg, Count, F, Q, QuerySet, Sum
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
from management.views.requests import STATUSES

TOTALS = {
    "requests": Count("id"),
    **{name: Count("id", filter=q) for name, q in STATUSES.items()},
    "input": Sum("input_tokens", default=0),
    "cached": Sum("cached_input_tokens", default=0),
    "output": Sum("output_tokens", default=0),
    "reasoning": Sum("reasoning_tokens", default=0),
}
"""The figures of a group of requests: counts per status and token sums."""
STEPS = {
    "minute": ("H:i", _("Tokens per minute"), _("Requests per minute")),
    "hour": ("H:00", _("Tokens per hour"), _("Requests per hour")),
    "day": ("j M", _("Tokens per day"), _("Requests per day")),
}
"""How the chart labels a time bucket of each size, and its titles."""
STATUS_ROWS = (
    ("ok", _("OK"), "2xx", "ok", "st-ok"),
    ("rate_limited", _("Rate-limited"), "429", "warning", "st-warn"),
    ("client_error", _("Client error"), "4xx", "off", "st-off"),
    ("server_error", _("Server error"), "5xx", "failed", "st-err"),
)
LOG_SPANS = {"1h": "1h", "1d": "1d", "1w": "7d", "1m": "30d"}
"""The Requests log's time filter for each time frame, for links into the log."""
AXIS_LABELS, SPARK_BARS, TOP_ROWS, FAILED_ALERT_PERCENT = 8, 12, 10, 2


def _bucket_config_for_range(
    start_time: dt.datetime, end_time: dt.datetime
) -> tuple[timedelta, Any]:
    """Determine bucket delta and round function for a custom time range."""
    duration = end_time - start_time
    if duration <= timedelta(hours=2):
        delta = timedelta(minutes=1)
        round_time = lambda d: d.replace(second=0, microsecond=0)
    elif duration <= timedelta(days=2):
        delta = timedelta(hours=1)
        round_time = lambda d: d.replace(minute=0, second=0, microsecond=0)
    else:
        delta = timedelta(days=1)
        round_time = lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0)
    return delta, round_time


def get_all_buckets(start_time: dt.datetime, now: dt.datetime, freq_label: str) -> list[int]:
    points = []
    if freq_label == "1h":
        # one point per minute
        delta = timedelta(minutes=1)
        round_time = lambda d: d.replace(second=0, microsecond=0)
    elif freq_label == "1d":
        # one point per hour
        delta = timedelta(hours=1)
        round_time = lambda d: d.replace(minute=0, second=0, microsecond=0)
    elif freq_label == "1m":
        # one point per day
        delta = timedelta(days=1)
        round_time = lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0)
    elif freq_label == "custom":
        delta, round_time = _bucket_config_for_range(start_time, now)
    else:
        # one point per day (1w fallback)
        delta = timedelta(days=1)
        round_time = lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0)

    d = round_time(start_time)
    end_time = round_time(now)
    while d <= end_time:
        points.append(int(d.timestamp() * 1000))
        d = d + delta
    return points


def _with_shares(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add each row's tokens, new input, and whether its failures deserve a red flag."""
    for row in rows:
        row["tokens"] = row["input"] + row["output"]
        row["new_input"] = max(row["input"] - row["cached"], 0)
        row["failing"] = 100 * row["failed"] > FAILED_ALERT_PERCENT * row["requests"]
    return rows


def _tokens(count: float) -> str:
    return ngettext("%(count)s token", "%(count)s tokens", count) % {"count": compact(count)}


def _requests(count: float) -> str:
    return ngettext("%(count)s request", "%(count)s requests", count) % {"count": intcomma(count)}


def _bars(
    groups: list[tuple[str, dict[str, Any]]],
    parts_of: Callable[[dict[str, Any]], tuple[tuple[str, str, float], ...]],
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
        span_choices = {
            "1h": {"delta": timedelta(hours=1), "trunc": TruncMinute("timestamp")},
            "1d": {"delta": timedelta(days=1), "trunc": TruncHour("timestamp")},
            "1w": {"delta": timedelta(days=7), "trunc": TruncDay("timestamp")},
            "1m": {"delta": timedelta(days=30), "trunc": TruncDay("timestamp")},
        }
        now = timezone.now()

        # Parse optional custom date range
        custom_start_str = self.request.GET.get("start", "")
        custom_end_str = self.request.GET.get("end", "")
        custom_start, custom_end, custom_error = self._parse_custom_range(
            custom_start_str, custom_end_str, now
        )

        selected_span = self.request.GET.get("span", "1d")
        if custom_start and custom_end:
            selected_span = "custom"
        elif selected_span not in span_choices:
            selected_span = "1d"

        if selected_span == "custom" and custom_start and custom_end:
            start_time = custom_start
            end_time = custom_end
            # Determine trunc based on range duration
            duration = end_time - start_time
            if duration <= timedelta(hours=2):
                trunc_fn = TruncMinute("timestamp")
            elif duration <= timedelta(days=2):
                trunc_fn = TruncHour("timestamp")
            else:
                trunc_fn = TruncDay("timestamp")
            span_cfg: dict[str, Any] = {"delta": duration, "trunc": trunc_fn}
        else:
            span_cfg = span_choices[selected_span]
            start_time = now - span_cfg["delta"]
            end_time = now

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

        # Figures in total and per time bucket
        totals = _with_shares([reqs_span.aggregate(**TOTALS)])[0]
        series = self._series(
            reqs_span, span_cfg["trunc"], get_all_buckets(start_time, end_time, selected_span)
        )

        # Top entities: orgs global, or tokens per org, or user IDs per token
        limit = 100 if self.request.GET.get("rows") == "all" else TOP_ROWS
        top_items = _with_shares(
            list(self._get_top_items(reqs_span, selected_org, selected_token)[: limit + 1])
        )
        openable = set(
            profile.visible_tokens()
            .filter(pk__in=[item["token_id"] for item in top_items if "token_id" in item])
            .values_list("pk", flat=True)
        )

        # LiteLLM treats a model without a mode as chat.
        modes = {
            m["model_name"]: (m.get("model_info") or {}).get("mode") or "chat"
            for m in router_models
        }
        per_model = _with_shares(
            list(
                reqs_span.values("model")
                .annotate(**TOTALS, avg_ms=Avg("response_time_ms"))
                .order_by()
            )
        )
        per_model.sort(key=lambda row: (-row["tokens"], -row["requests"], row["model"]))

        # Links into the Requests log under the same filters
        log_filters = {
            "span": LOG_SPANS.get(selected_span),
            "key": selected_token.pk if selected_token else None,
            "model": selected_model,
        }

        def log_url(**params: str) -> str:
            query = {name: v for name, v in {**log_filters, **params}.items() if v}
            return f"{reverse('requests')}?{urlencode(query)}"

        for row in per_model:
            row["mode"] = modes.get(row["model"])
            row["log_url"] = log_url(model=row["model"])

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
                "charts": self._charts(series, span_cfg["trunc"].kind),
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
                "per_model": per_model,
                "models_busiest": max((row["tokens"] for row in per_model), default=0),
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
                    if sel and sel.isdigit() and is_global_admin
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
        selected_token = None
        if token_param and token_param.isdigit():
            try:
                tok = Token.objects.get(pk=int(token_param))
                # Ensure the token belongs to the selected_org if scoped
                if selected_org is not None:
                    tok_org_id = tok.user.profile.org_id if tok.user else None
                    sa_org_id = tok.service_account.team.org_id if tok.service_account else None
                    if selected_org.id not in (tok_org_id, sa_org_id):
                        raise Token.DoesNotExist
                selected_token = tok
            except Token.DoesNotExist:
                selected_token = None
        return selected_token

    @staticmethod
    def _series(reqs: QuerySet[Request], trunc: Any, buckets: list[int]) -> list[dict[str, Any]]:
        """The figures of each time bucket, labelled for the charts."""
        found = {
            int(row.pop("period").timestamp() * 1000): row
            for row in reqs.annotate(period=trunc).values("period").annotate(**TOTALS).order_by()
        }
        return _with_shares(
            [
                {
                    **found.get(ms, dict.fromkeys(TOTALS, 0)),
                    "label": date_format(
                        timezone.localtime(dt.datetime.fromtimestamp(ms / 1000, dt.UTC)),
                        STEPS[trunc.kind][0],
                    ),
                }
                for ms in buckets
            ]
        )

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
            spark, _ = _bars(slots, lambda figures: (("i", "", figures[field]),), show)
            return {"label": label, "value": value, "note": note, "spark": spark}

        share = percent(totals["cached"], totals["input"])
        return [
            card(gettext("Requests"), intcomma(totals["requests"]), "", "requests", _requests),
            card(
                gettext("Input tokens"),
                compact(totals["input"]),
                gettext("%(share)s from cache") % {"share": share} if totals["input"] else "",
                "input",
                _tokens,
            ),
            card(
                gettext("Output tokens"),
                compact(totals["output"]),
                gettext("%(count)s reasoning") % {"count": compact(totals["reasoning"])}
                if totals["reasoning"]
                else "",
                "output",
                _tokens,
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
    def _charts(rows: list[dict[str, Any]], step: str) -> dict[str, Any]:
        """The tokens and the requests chart, a stacked column per time bucket, and their axis."""
        groups = [(row["label"], row) for row in rows]
        tokens, most_tokens = _bars(
            groups,
            lambda row: (
                ("c", gettext("From cache"), row["cached"]),
                ("n", gettext("New input"), row["new_input"]),
                ("o", gettext("Output"), row["output"]),
            ),
            _tokens,
        )
        requests, most_requests = _bars(
            groups,
            lambda row: (
                ("i", gettext("OK"), row["requests"] - row["failed"] - row["rate_limited"]),
                ("r", gettext("Rate-limited"), row["rate_limited"]),
                ("f", gettext("Failed"), row["failed"]),
            ),
            _requests,
        )
        every = math.ceil(len(rows) / AXIS_LABELS) or 1
        return {
            "columns": len(rows),
            "axis": [
                {"label": row["label"], "column": index + 1, "span": min(every, len(rows) - index)}
                for index, row in enumerate(rows)
                if index % every == 0
            ],
            "metrics": [
                ("tokens", STEPS[step][1], _tokens(most_tokens), tokens),
                ("requests", STEPS[step][2], _requests(most_requests), requests),
            ],
        }

    def _get_top_items(
        self, reqs: QuerySet[Request], selected_org: Org | None, selected_token: Token | None
    ) -> QuerySet[Request, dict[str, Any]]:
        """Annotate the `Request` queryset and get the top-level entities as dicts"""
        if selected_org is None:
            # Top orgs sorted by tokens - only available for global admins.
            # Note: service account tokens belong to the same org as the token's user.
            top_items = reqs.annotate(
                name=F("token__user__profile__org__name"), org_id=F("token__user__profile__org__id")
            ).values("name", "org_id")
        elif selected_token is None:
            # Top tokens for the selected org by tokens
            top_items = reqs.values("token_id").annotate(
                name=F("token__name"),
                owner_id=F("token__user_id"),
                user_email=F("token__user__email"),
                service_account_name=F("token__service_account__name"),
                team_name=F("token__service_account__team__name"),
            )
        else:
            # Top user IDs for the selected token by tokens
            top_items = reqs.annotate(name=F("token__name")).values("name", "user_id")

        # Add the figures and ordering common to all cases
        return top_items.annotate(
            **TOTALS, total_sum=Sum(F("input_tokens") + F("output_tokens"))
        ).order_by("-total_sum", "-requests")
