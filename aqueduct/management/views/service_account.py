from django.contrib import messages
from django.db import transaction
from django.db.models import QuerySet
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.translation import gettext
from django.views import View
from django.views.generic import CreateView, DeleteView, UpdateView

from management.access import TeamAct
from management.deletion import Crumb, deletion
from management.forms import MemberEmailForm, ServiceAccountForm
from management.keys import key_report
from management.models import ServiceAccount, Token, UserProfile
from management.views.base import BaseServiceAccountView, BaseTeamView, show_new_key

OWNER_CHOICES = 50


class ServiceAccountCreateView(BaseTeamView, CreateView):
    model = ServiceAccount
    form_class = ServiceAccountForm
    template_name = "management/create/service_account.html"
    team_act = TeamAct.MANAGE

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["team"] = self.team
        return kwargs

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["team"] = self.team
        context["view_title"] = gettext("Add a service account")
        context["cancel_url"] = reverse("team", kwargs={"id": self.team.id})
        return context

    @transaction.atomic
    def form_valid(self, form) -> HttpResponse:
        form.instance.team = self.team
        self.object = form.save()
        token = Token(
            name=self.object.name,
            user=self.request.user,
            service_account=self.object,
            expires_at=form.cleaned_data.get("token_expires_at"),
        )
        secret_key = token._set_new_key()
        token.save()
        show_new_key(self.request, token.id, secret_key)
        return redirect("token_detail", id=token.id)


class ServiceAccountDeleteView(BaseServiceAccountView, DeleteView):
    template_name = "management/common/confirm_delete.html"

    def get_success_url(self) -> str:
        return reverse("team", kwargs={"id": self.team.id})

    def get_context_data(self, **kwargs) -> dict[str, object]:
        token = Token.objects.filter(service_account=self.object).first()
        return super().get_context_data(
            confirmation=deletion(self.object),
            activity=key_report(token, timezone.now()).totals if token else None,
            cancel_url=self.get_success_url(),
            trail=[
                Crumb(gettext("Teams"), reverse("org")),
                Crumb(self.team.name, self.get_success_url()),
                Crumb(self.object.name, mono=True),
            ],
            **kwargs,
        )

    def form_valid(self, form) -> HttpResponse:
        name = self.object.name
        response = super().form_valid(form)
        messages.success(
            self.request, gettext("Service account %(name)s deleted.") % {"name": name}
        )
        return response


class ServiceAccountUpdateView(BaseServiceAccountView, UpdateView):
    context_object_name = "service_account"
    form_class = ServiceAccountForm
    template_name = "management/edit/service_account.html"

    def get_initial(self) -> dict:
        initial = super().get_initial()
        token = Token.objects.filter(service_account=self.get_object()).first()
        if token is not None and token.expires_at:
            initial["token_expires_at"] = token.expires_at.strftime("%Y-%m-%dT%H:%M")
        return initial

    def get_form_kwargs(self) -> dict:
        kwargs = super().get_form_kwargs()
        kwargs["team"] = self.team
        return kwargs

    def form_valid(self, form) -> HttpResponse:
        response = super().form_valid(form)
        token = Token.objects.filter(service_account=self.object).first()
        if token is not None:
            token.expires_at = form.cleaned_data.get("token_expires_at")
            token.save(update_fields=["expires_at"])
        messages.success(
            self.request, gettext("Service account %(name)s saved.") % {"name": self.object.name}
        )
        return response

    def get_success_url(self) -> str:
        return reverse("team", kwargs={"id": self.team.id})

    def get_context_data(self, **kwargs) -> dict[str, object]:
        context = super().get_context_data(**kwargs)
        context["view_title"] = gettext("Edit service account")
        context["cancel_url"] = self.get_success_url()
        context["team"] = self.team
        return context


class ServiceAccountTransferOwnershipView(BaseServiceAccountView, View):
    template_name = "management/transfer_service_account.html"

    def get_success_url(self) -> str:
        return reverse("team", kwargs={"id": self.team.id})

    def owners(self) -> QuerySet[UserProfile]:
        # A switched platform admin or a leftover membership keeps a profile in another org.
        return self.team.member_profiles.filter(org=self.team.org)

    def owner_form(self, data=None) -> MemberEmailForm:
        team = self.team
        return MemberEmailForm(
            data,
            candidates=self.owners(),
            not_found=lambda email: (
                gettext("%(email)s is not a member of %(team)s.")
                % {"email": email, "team": team.name}
            ),
        )

    def render_form(self, form: MemberEmailForm, token: Token | None) -> HttpResponse:
        team = self.team
        owner = token.user if token else None
        others = self.owners().exclude(user=owner)
        choices = None
        if self.owners().count() <= OWNER_CHOICES:
            choices = list(others.order_by("user__email").values_list("user__email", flat=True))
        context = {
            "service_account": self.get_object(),
            "team": team,
            "form": form,
            "current_owner_email": owner.email if owner else None,
            "has_other_members": others.exists(),
            "owner_choices": choices,
            "cancel_url": self.get_success_url(),
            "view_title": gettext("Transfer ownership"),
        }
        return TemplateResponse(request=self.request, template=self.template_name, context=context)

    def get(self, request, *args, **kwargs) -> HttpResponse:
        token = Token.objects.filter(service_account=self.get_object()).select_related("user")
        return self.render_form(self.owner_form(), token.first())

    @method_decorator(transaction.atomic)
    def post(self, request, *args, **kwargs) -> HttpResponse:
        sa = self.get_object()
        token = Token.objects.select_for_update().filter(service_account=sa).first()
        if token is None:
            messages.error(
                request,
                gettext("Service account %(name)s has no key to transfer.") % {"name": sa.name},
            )
            return redirect(self.get_success_url())
        form = self.owner_form(request.POST)
        if form.is_valid() and form.profile.user_id == token.user_id:
            form.add_error(
                "email",
                gettext("%(email)s already owns this key.") % {"email": form.profile.user.email},
            )
        if not form.is_valid():
            return self.render_form(form, token)
        token.user = form.profile.user
        token.save(update_fields=["user"])
        messages.success(
            request,
            gettext("%(email)s now owns the key of %(name)s.")
            % {"email": form.profile.user.email, "name": sa.name},
        )
        return redirect(self.get_success_url())
