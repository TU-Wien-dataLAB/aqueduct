import json
from datetime import timedelta

from django.db.models import Count, Sum, Avg, Q, F
from django.db.models.functions import TruncHour, TruncDay, Coalesce, TruncMinute
from django.utils import timezone
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from gateway.config import get_router_config
from .base import BaseAqueductView
from ..models import Request, Token, Org


def get_all_buckets(start_time, now, freq_label):
    points = []
    if freq_label == "1h":
        # one point per minute
        delta = timedelta(minutes=1)
        get_next = lambda d: d + delta
        round_time = lambda d: d.replace(second=0, microsecond=0)
    elif freq_label == "1d":
        # one point per hour
        delta = timedelta(hours=1)
        get_next = lambda d: d + delta
        round_time = lambda d: d.replace(minute=0, second=0, microsecond=0)
    else:
        # one point per day
        delta = timedelta(days=1)
        get_next = lambda d: d + delta
        round_time = lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0)
    d = round_time(start_time)
    end_time = round_time(now)
    while d <= end_time:
        points.append(int(d.timestamp() * 1000))
        d = get_next(d)
    return points


class UsageDashboardView(BaseAqueductView, TemplateView):
    """
    Usage dashboard page: switch between all orgs (global) and per-org token usage.
    """
    template_name = 'management/usage.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.profile
        is_global_admin = profile.is_admin() if profile else False

        # Organization selection: 'all' for global, or org id
        orgs = Org.objects.all() if is_global_admin else Org.objects.filter(pk=profile.org_id)
        sel = self.request.GET.get('org')
        if sel == 'all' and is_global_admin:
            selected_org = None
        else:
            try:
                selected_org = Org.objects.get(
                    pk=int(sel)) if sel and sel.isdigit() and is_global_admin else profile.org
            except Org.DoesNotExist:
                selected_org = self.org

        # Base queryset of requests in scope
        # Global admins see all requests; org admins see all for the selected org;
        # standard users see only their own tokens or service accounts of teams they belong to
        if selected_org is None:
            reqs = Request.objects.all()
        else:
            # Org-level access: full org view for org admins
            if profile.is_org_admin(selected_org):
                reqs = Request.objects.filter(
                    Q(token__user__profile__org=selected_org)
                    | Q(token__service_account__team__org=selected_org)
                )
            else:
                # Standard user: restrict to own user tokens or service accounts in their teams
                user = profile.user
                teams = self.get_teams_for_user()
                reqs = Request.objects.filter(
                    Q(token__user=user)
                    | Q(token__service_account__team__in=teams)
                )

        # Only include requests for models configured in the router
        allowed_models = [m['model_name'] for m in get_router_config().get('model_list', [])]
        reqs = reqs.filter(model__in=allowed_models)

        # Optional token selection: filter to a single token if provided
        token_param = self.request.GET.get('token')
        selected_token = None
        if token_param and token_param.isdigit():
            try:
                tok = Token.objects.get(pk=int(token_param))
                # Ensure the token belongs to the selected_org if scoped
                if selected_org is not None:
                    tok_org_id = tok.user.profile.org_id if tok.user else None
                    sa_org_id = tok.service_account.team.org_id if tok.service_account else None
                    if tok_org_id != selected_org.id and sa_org_id != selected_org.id:
                        raise Token.DoesNotExist
                selected_token = tok
                reqs = reqs.filter(token=selected_token)
            except Token.DoesNotExist:
                selected_token = None

        # Time frame selection: 1 day, 1 week, or 1 month
        span_choices = {
            '1h': {'delta': timedelta(hours=1), 'trunc': TruncMinute('timestamp')},
            '1d': {'delta': timedelta(days=1), 'trunc': TruncHour('timestamp')},
            '1w': {'delta': timedelta(days=7), 'trunc': TruncDay('timestamp')},
        }
        selected_span = self.request.GET.get('span', '1d')
        if selected_span not in span_choices:
            selected_span = '1d'
        span_cfg = span_choices[selected_span]

        now = timezone.now()
        # Filter requests based on selected time span
        start_time = now - span_cfg['delta']
        reqs_span = reqs.filter(timestamp__gte=start_time)

        # Build time-series data for chart
        qs = (
            reqs_span
            .annotate(period=span_cfg['trunc'])
            .values('period')
            .annotate(count=Count('id'))
            .order_by('period')
        )
        period_to_count = {int(item['period'].timestamp() * 1000): item['count'] for item in qs}
        buckets = get_all_buckets(start_time, now, selected_span)
        timeseries = [[b, period_to_count.get(b, 0)] for b in buckets]

        # Top entities: tokens per org or orgs global
        if selected_org is None:
            # Top orgs by request count (up to 100)
            agg = (
                reqs_span.annotate(
                    org_id=Coalesce(
                        F('token__user__profile__org_id'),
                        F('token__service_account__team__org_id')
                    ),
                )
                .values('org_id')
                .annotate(count=Count('id'))
                .order_by('-count')[:100]
            )
            org_ids = [o['org_id'] for o in agg if o['org_id'] is not None]
            org_objs = {org.id: org for org in Org.objects.filter(id__in=org_ids)}
            top_items = [
                {'object': org_objs.get(o['org_id']), 'count': o['count']}
                for o in agg
                if o['org_id'] in org_objs
            ]
        else:
            # Top tokens by request count
            agg = (
                reqs_span.values('token__id')
                .annotate(count=Count('id'))
                .order_by('-count')[:100]
            )
            token_ids = [t['token__id'] for t in agg if t['token__id'] is not None]
            token_objs = {tok.id: tok for tok in
                          Token.objects.filter(id__in=token_ids).select_related('user', 'service_account')}
            top_items = [
                {'object': token_objs.get(t['token__id']), 'count': t['count']}
                for t in agg
                if t['token__id'] in token_objs
            ]

        # Simple statistics
        total_requests = reqs_span.count()
        failed_requests = reqs_span.filter(status_code__gte=400).count()

        per_model = list(
            reqs_span.values('model')
            .annotate(count=Count('id'))
            .order_by('-count')
        )
        present_models = {m['model'] for m in per_model}
        missing_models = [{"model": m['model_name'], "count": 0} for m in get_router_config()['model_list'] if
                          m['model_name'] not in present_models]
        per_model += missing_models

        avg_time_comp = reqs_span.filter(path__icontains='completion').aggregate(avg=Avg('response_time_ms'))['avg']
        avg_time_emb = reqs_span.filter(path__icontains='embedding').aggregate(avg=Avg('response_time_ms'))['avg']
        tokens_sum = reqs_span.aggregate(
            input_sum=Sum('input_tokens'), output_sum=Sum('output_tokens')
        )
        input_tokens = tokens_sum.get('input_sum') or 0
        output_tokens = tokens_sum.get('output_sum') or 0

        context.update({
            'title': 'Usage',
            'is_global_admin': is_global_admin,
            'orgs': orgs,
            'selected_org': selected_org,
            'selected_token': selected_token,
            'span_choices': list(span_choices.keys()),
            'selected_span': selected_span,
            'timeseries': json.dumps(timeseries),
            'top_items': top_items,
            'top_k_choices': [10, 25, 100],
            'total_requests': total_requests,
            'failed_requests': failed_requests,
            'per_model_json': json.dumps(per_model),
            'avg_time_completion': avg_time_comp or 0,
            'avg_time_embedding': avg_time_emb or 0,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        })
        return context
