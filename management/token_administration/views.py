from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.conf import settings
from django.views.generic import TemplateView
from django.shortcuts import redirect, render, get_object_or_404

from .models import Team


class SSOTemplateView(TemplateView):
    template_name = 'token_administration/login.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            # Redirect the logged-in user to the desired page
            return redirect('/')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add setting with fallback default
        context['OIDC_PROVIDER'] = getattr(settings, 'OIDC_PROVIDER', 'SSO')
        return context


@login_required
def tokens(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@login_required
def org(request):
    # TODO: only show teams user is part of if group is 'team-admin' or 'user'
    return render(request, 'token_administration/org.html', context={})


@login_required
def team_create(request):
    # Add logic here to handle the team creation form
    # return render(request, 'your_template_directory/team_create.html')
    # TODO: only org-admin can create teams
    return HttpResponse("Team Creation")


@login_required
def team(request, id: int):
    t = get_object_or_404(Team, id=id)
    # TODO: only show team if user is part of it if group is 'team-admin' or 'user'
    context = {'team': t}
    return render(request, 'token_administration/team.html', context)
