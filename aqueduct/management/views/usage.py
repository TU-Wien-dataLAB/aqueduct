from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.http import urlencode
from django.utils.translation import gettext
from django.views.generic import TemplateView

from management import usage_report as report
from management.access import EveryOrg, KeyAct, Viewer, WholeOrg
from management.charts import OVERVIEW_STATS, columns, slots, stats
from management.ids import parse_id
from management.model_catalog import model_kinds
from management.models import Token, UserProfile
from management.org_choice import viewer_of
from management.periods import DEFAULT_SPAN
from management.request_filters import Filters, Key, describe_key, key_search, read_filters
from management.request_paging import PAGE_SIZE, Header
from management.usage_source import UsageSource


def _viewer(request: HttpRequest) -> Viewer:
    viewer = viewer_of(request)
    if viewer is None:
        raise UserProfile.DoesNotExist
    return viewer


def _describe(filters: Filters) -> Callable[[Token], Key]:
    every_org = isinstance(filters.viewer.chosen, EveryOrg)

    def describe(token: Token) -> Key:
        key = describe_key(token, filters.viewer, with_org=every_org)
        if not filters.can_pick_scope:
            return replace(key, note="")
        return replace(key, note=gettext("You")) if key.own else key

    return describe


class UsageDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "management/usage.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        viewer = _viewer(self.request)
        params = self.request.GET
        now = timezone.now()
        filters = read_filters(viewer, params, default_span=DEFAULT_SPAN, now=now)
        selected_key, period = filters.key, filters.period
        chosen = viewer.chosen
        selected_org = chosen.org if isinstance(chosen, WholeOrg) else None

        retention_days = settings.REQUEST_RETENTION_DAYS
        source = UsageSource.of(filters)
        timeline = source.series()
        total = timeline.total

        more_keys = False
        if selected_key is not None:
            keys = report.user_id_shares(period.of(filters.requests), total.tokens)
        elif filters.only_key is not None:
            keys = ()
        else:
            ranked = report.ranked(
                source.key_totals(), report.KeyOrder.TOKENS, viewer.keys(KeyAct.SEE)
            )
            more_keys = len(ranked) > report.TOP_KEYS
            keys = report.key_shares(ranked[: report.TOP_KEYS], total.tokens, _describe(filters))

        context.update(
            {
                "filters": filters,
                "period": period,
                "kpis": stats(slots(timeline.points, period.step), OVERVIEW_STATS),
                "requests_in_period": total.requests,
                "charts": tuple(
                    (shown, columns(timeline.points, period.step, shown))
                    for shown in ("tokens", "requests")
                ),
                "statuses": report.statuses(timeline, filters.log_url),
                "models": report.model_rows(source.models(), model_kinds(), filters.log_url),
                "keys": keys,
                "more_keys": more_keys,
                "all_keys_url": f"{reverse('usage_keys')}?{urlencode(filters.query)}",
                "orgs_usage": ()
                if selected_org is not None
                else report.org_shares(source.org_tokens()),
                "retention_warning": period.start < now - timedelta(days=retention_days),
                "retention_days": int(retention_days),
            }
        )
        return context


class KeyUsageView(LoginRequiredMixin, TemplateView):
    template_name = "management/usage_keys.html"

    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        params = {name: value for name, value in request.GET.items() if name != "key"}
        filters = read_filters(
            _viewer(request), params, default_span=DEFAULT_SPAN, now=timezone.now()
        )
        only_key = filters.only_key
        if not filters.can_pick_scope and only_key is not None and only_key.href is not None:
            return redirect(only_key.href)
        return self.render_to_response(self.get_context_data(filters=filters, **kwargs))

    def get_context_data(self, *, filters: Filters, **kwargs: Any) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        params = self.request.GET
        q = params.get("q", "").replace("\x00", "").strip()
        search = key_search(q, by_owner=filters.can_pick_scope)
        order = report.KeyOrder.parse(params.get("sort", "")) or report.KeyOrder.TOKENS
        totals = UsageSource.of(filters).key_totals()
        everyone = report.KeyTotals.of(totals)
        tokens = filters.viewer.keys(KeyAct.SEE)
        if q:
            ids = set(tokens.filter(search).values_list("id", flat=True))
            totals = {token_id: a for token_id, a in totals.items() if token_id in ids}
        rows = report.ranked(totals, order, tokens)

        pages = Paginator(rows, PAGE_SIZE).get_page(parse_id(params.get("page")) or 1)
        keys = report.key_shares(
            pages.object_list,
            everyone.tokens,
            _describe(filters),
            everyone.busiest,
            limits_at=timezone.now(),
        )
        context.update(
            {
                "filters": filters,
                "usage_url": f"{reverse('usage')}?{urlencode(filters.query)}",
                "keys": keys,
                "count": len(rows),
                "q": q,
                "order": order,
                "headers": {
                    option.value: Header(
                        sortable=True,
                        choice=None if option is report.KeyOrder.TOKENS else option.value,
                        aria_sort=(
                            ("ascending" if option.ascending else "descending")
                            if option is order
                            else None
                        ),
                    )
                    for option in report.KeyOrder
                },
                "pages": pages,
            }
        )
        return context
