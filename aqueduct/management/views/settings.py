from django.conf import settings
from django.views.generic import TemplateView
from tos.models import UserAgreement

from management.views.base import BaseAqueductView


class SettingsView(BaseAqueductView, TemplateView):
    """The viewer's account: org, role, teams, personal keys and the accepted terms."""

    template_name = "management/settings.html"

    def get_context_data(self, **kwargs) -> dict[str, object]:
        profile = self.profile
        agreement = (
            UserAgreement.objects.filter(user=self.request.user, terms_of_service__active=True)
            .order_by("-created")
            .first()
            if settings.TOS_ENABLED
            else None
        )
        return super().get_context_data(
            memberships=profile.teammembership_set.filter(team__in=profile.member_teams())
            .select_related("team")
            .order_by("team__name"),
            personal_keys=profile.own_tokens().count(),
            max_tokens=settings.MAX_USER_TOKENS,
            terms_accepted_on=agreement.created if agreement else None,
            **kwargs,
        )
