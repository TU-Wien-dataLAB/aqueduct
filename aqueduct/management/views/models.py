from typing import Any

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

from management.access import TeamAct
from management.model_catalog import Category, load_catalog, scopes_for
from management.nav import gateway_url
from management.org_choice import viewer_of


class ModelListView(LoginRequiredMixin, TemplateView):
    template_name = "management/model_list.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        viewer = viewer_of(self.request)
        scopes = scopes_for(viewer.profile, viewer.teams(TeamAct.MINE)) if viewer else ()
        params = self.request.GET
        catalog = load_catalog(scopes)
        category = Category.parse(params.get("type", ""))
        query = params.get("q", "").strip()
        cards = catalog.search(category, query)
        selected = catalog.find(params.get("model", "")) or next(iter(cards or catalog.cards), None)
        context.update(
            catalog=catalog,
            category=category,
            query=query,
            cards=cards,
            selected=selected,
            snippet=selected.curl(gateway_url(self.request)) if selected else None,
            show_served_model=viewer is not None and viewer.administers(viewer.org),
        )
        return context
