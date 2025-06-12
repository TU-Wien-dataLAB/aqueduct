import json
from datetime import timedelta

from django.db.models import Count, Sum, Avg, Q, F
from django.db.models.functions import TruncHour, TruncDay, Coalesce, TruncMinute
from django.utils import timezone
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from ..models import Request, Token, Org


class UsageDashboardView(LoginRequiredMixin, TemplateView):
    """
    Usage dashboard page: switch between all orgs (global) and per-org token usage.
    """
    template_name = 'management/usage.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        profile = getattr(user, 'profile', None)
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
                selected_org = profile.org

        # Base queryset of requests in scope
        if selected_org is None:
            reqs = Request.objects.all()
        else:
            reqs = Request.objects.filter(
                Q(token__user__profile__org=selected_org)
                | Q(token__service_account__team__org=selected_org)
            )

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
        timeseries = [
            [int(item['period'].timestamp() * 1000), item['count']]
            for item in qs
        ]

        # Top entities: tokens per org or orgs global
        if selected_org is None:
            # Top orgs by request count (up to 100)
            agg = (
                reqs.annotate(
                    org_id=Coalesce(
                        F('token__user__profile__org_id'),
                        F('token__service_account__team__org_id')
                    ),
                    org_name=Coalesce(
                        F('token__user__profile__org__name'),
                        F('token__service_account__team__org__name')
                    )
                )
                .values('org_id', 'org_name')
                .annotate(count=Count('id'))
                .order_by('-count')[:100]
            )
            top_items = [{'id': o['org_id'], 'name': o['org_name'], 'count': o['count']} for o in agg]
        else:
            # Top tokens by request count
            agg = (
                reqs.values('token__id', 'token__name')
                .annotate(count=Count('id'))
                .order_by('-count')[:100]
            )
            top_items = [
                {'id': t['token__id'], 'name': t['token__name'], 'count': t['count']}
                for t in agg
            ]

        # Simple statistics
        total_requests = reqs_span.count()
        failed_requests = reqs_span.filter(status_code__gte=400).count()
        per_model = (
            reqs_span.values('model')
            .annotate(count=Count('id'))
            .order_by('-count')
        )
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
            'span_choices': list(span_choices.keys()),
            'selected_span': selected_span,
            'timeseries': json.dumps(timeseries),
            'top_items': top_items,
            'top_items_json': json.dumps(top_items),
            'top_k_choices': [10, 25, 100],
            'total_requests': total_requests,
            'failed_requests': failed_requests,
            'per_model_json': json.dumps(list(per_model)),
            'avg_time_completion': avg_time_comp or 0,
            'avg_time_embedding': avg_time_emb or 0,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        })
        return context
