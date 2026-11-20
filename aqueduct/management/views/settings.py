from datetime import datetime

from django.conf import settings
from django.contrib.auth.models import AbstractBaseUser
from django.views.generic import TemplateView
from tos.models import UserAgreement

from management.access import KeyAct
from management.nav import mounted
from management.views.base import BaseAqueductView


def accepted_terms_on(user: AbstractBaseUser) -> datetime | None:
    agreement = UserAgreement.objects.filter(user=user, terms_of_service__active=True).first()
    return agreement.created if agreement else None


class SettingsView(BaseAqueductView, TemplateView):
    template_name = "management/settings.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        profile = self.viewer.profile
        show_terms = mounted("tos")
        context.update(
            memberships=profile.teammembership_set.filter(team_id__in=self.viewer.member_of)
            .select_related("team")
            .order_by("team__name"),
            personal_keys=self.viewer.keys(KeyAct.OWN).count(),
            max_tokens=settings.MAX_USER_TOKENS,
            show_terms=show_terms,
            terms_accepted_on=accepted_terms_on(self.request.user) if show_terms else None,
        )
        return context
