from collections.abc import Callable
from dataclasses import dataclass

from django.conf import settings
from django.contrib import admin
from django.http import HttpRequest
from django.urls import NoReverseMatch, reverse
from django.utils.functional import Promise
from django.utils.translation import gettext, ngettext
from django.utils.translation import gettext_lazy as _
from silk.config import SilkyConfig

from management.access import WholeOrg
from management.icons import Icon
from management.ids import parse_id
from management.models import Org, ServiceAccount, Team
from management.org_choice import viewer_of

Access = Callable[[HttpRequest], bool]


def everyone(request: HttpRequest) -> bool:
    return True


def signed_in(request: HttpRequest) -> bool:
    return request.user.is_authenticated


def django_admin(request: HttpRequest) -> bool:
    return admin.site.has_permission(request)


def org_admin(request: HttpRequest) -> bool:
    viewer = viewer_of(request)
    return viewer is not None and viewer.administers(viewer.org)


def mounted(url_name: str) -> bool:
    try:
        reverse(url_name)
    except NoReverseMatch:
        return False
    return True


def silk_profiling(request: HttpRequest) -> bool:
    return SilkyConfig().SILKY_PERMISSIONS(request.user)


@dataclass(frozen=True)
class Page:
    label: str | Promise
    url_name: str
    icon: Icon
    access: Access = signed_in
    url_families: tuple[str, ...] = ()
    external: bool = False

    def is_active(self, view_name: str) -> bool:
        return any(
            view_name == family or view_name.startswith((f"{family}_", f"{family}:"))
            for family in self.url_families or (self.url_name,)
        )


@dataclass(frozen=True)
class Group:
    title: str | Promise | None
    pages: tuple[Page, ...]
    teams: bool = False
    """The group lists the teams the viewer is in, and is left out for a person in none."""


NAV: tuple[Group, ...] = (
    Group(
        None,
        (
            Page(_("API keys"), "tokens", Icon.KEY, url_families=("tokens", "token")),
            Page(_("Requests"), "requests", Icon.REQUESTS),
            Page(_("Usage"), "usage", Icon.OVERVIEW),
            Page(_("Models"), "models", Icon.MODELS),
            Page(_("MCP servers"), "mcp_servers", Icon.MCP),
        ),
    ),
    Group(None, (), teams=True),
    Group(
        _("Organization"),
        (
            Page(_("All keys"), "usage_keys", Icon.KEY, access=org_admin),
            Page(
                _("Teams"),
                "org",
                Icon.TEAMS,
                access=org_admin,
                url_families=("org", "team", "service_account"),
            ),
            Page(_("Limits and access"), "organization", Icon.ORGANIZATION, access=org_admin),
        ),
    ),
    Group(
        _("Resources"),
        (
            Page(_("Files"), "files", Icon.FILE, url_families=("files", "file")),
            Page(_("Batches"), "batches", Icon.BATCH),
            Page(
                _("Vector stores"),
                "vector_stores",
                Icon.VECTOR_STORE,
                url_families=("vector_stores", "vector_store"),
            ),
        ),
    ),
    Group(
        _("Admin"),
        (
            Page(
                _("Admin panel"),
                "admin:index",
                Icon.ADMIN,
                access=django_admin,
                url_families=("admin",),
                external=True,
            ),
            Page(
                _("Profiling"),
                "silk:summary",
                Icon.PROFILING,
                access=silk_profiling,
                url_families=("silk",),
                external=True,
            ),
        ),
    ),
)
NAV_BOTTOM: tuple[Page, ...] = (Page(_("Settings"), "settings", Icon.SETTINGS),)
HELP: tuple[Page, ...] = (Page(_("Terms of service"), "tos", Icon.FILE, access=everyone),)


@dataclass(frozen=True)
class Link:
    label: str | Promise
    href: str
    icon: Icon
    active: bool = False
    external: bool = False


@dataclass(frozen=True)
class Section:
    title: str | Promise | None
    links: tuple[Link, ...]


@dataclass(frozen=True)
class Viewer:
    email: str
    role: str | None


@dataclass(frozen=True)
class OrgSwitch:
    orgs: tuple[Org, ...]
    chosen: Org | None


@dataclass(frozen=True)
class Shell:
    sections: tuple[Section, ...]
    bottom: tuple[Link, ...]
    help_links: tuple[Link, ...]
    contact_links: tuple[Link, ...]
    viewer: Viewer | None
    gateway_url: str
    org_switch: OrgSwitch | None
    scope: str
    """Whose data an org-scoped page shows: the organization, or all of them."""

    @property
    def current(self) -> Link | None:
        nav = (link for section in self.sections for link in section.links)
        return next((link for link in (*nav, *self.bottom, *self.help_links) if link.active), None)


def gateway_url(request: HttpRequest) -> str:
    return f"{request.scheme}://{request.get_host()}/v1"


def _team_links(request: HttpRequest) -> tuple[Link, ...]:
    viewer = viewer_of(request)
    if viewer is None or not viewer.member_of:
        return ()
    match = request.resolver_match
    kwargs = match.kwargs if match else {}
    here: object = None
    if match and match.view_name.startswith("team"):
        here = parse_id(str(kwargs.get("id")))
    elif match and match.view_name.startswith("service_account"):
        here = (
            ServiceAccount.objects.filter(pk=parse_id(str(kwargs.get("service_account_id"))))
            .values_list("team_id", flat=True)
            .first()
        )
    teams = Team.objects.filter(pk__in=viewer.member_of).order_by("name")
    return tuple(
        Link(team.name, reverse("team", args=[team.pk]), Icon.TEAMS, team.pk == here)
        for team in teams
    )


def build_shell(request: HttpRequest) -> Shell:
    view_name = request.resolver_match.view_name if request.resolver_match else ""
    allowed = {
        page
        for group in (*NAV, Group(None, NAV_BOTTOM), Group(None, HELP))
        for page in group.pages
        if mounted(page.url_name) and page.access(request)
    }
    team_links = _team_links(request)
    # The most specific entry wins: All keys over Usage, a team of one's own over Teams.
    exact = any(page.url_name == view_name for page in allowed) or any(
        link.active for link in team_links
    )

    def visible(pages: tuple[Page, ...]) -> tuple[Link, ...]:
        return tuple(
            Link(
                page.label,
                reverse(page.url_name),
                page.icon,
                page.url_name == view_name if exact else page.is_active(view_name),
                page.external,
            )
            for page in pages
            if page in allowed
        )

    extra = tuple(
        Link(gettext(name) if name else name, url, Icon.LINK, external=True)
        for name, url in settings.EXTRA_NAV_LINKS.items()
    )
    return Shell(
        sections=tuple(
            Section(_team_title(team_links) if group.teams else group.title, links)
            for group in NAV
            if (links := team_links if group.teams else visible(group.pages))
        ),
        bottom=visible(NAV_BOTTOM),
        help_links=extra + visible(HELP),
        contact_links=tuple(
            Link(label, url, icon, external=True)
            for label, url, icon in (
                (gettext("Matrix support channel"), settings.MATRIX_SUPPORT_URL, Icon.MATRIX),
                (gettext("GitHub repository"), settings.GITHUB_REPOSITORY_URL, Icon.GITHUB),
            )
            if url
        ),
        viewer=_viewer(request),
        gateway_url=gateway_url(request),
        org_switch=_org_switch(request),
        scope=_scope(request),
    )


def _scope(request: HttpRequest) -> str:
    viewer = viewer_of(request)
    if viewer is None:
        return ""
    return viewer.chosen.name


def _team_title(links: tuple[Link, ...]) -> str:
    return ngettext("Team", "Teams", len(links))


def _org_switch(request: HttpRequest) -> OrgSwitch | None:
    viewer = viewer_of(request)
    if viewer is None or not viewer.can_switch_org:
        return None
    chosen = viewer.chosen.org if isinstance(viewer.chosen, WholeOrg) else None
    return OrgSwitch(tuple(viewer.orgs().order_by("name")), chosen)


def _viewer(request: HttpRequest) -> Viewer | None:
    user = request.user
    if not user.is_authenticated:
        return None
    viewer = viewer_of(request)
    if viewer is None:
        return Viewer(email=user.email, role=None)
    return Viewer(email=user.email, role=str(viewer.rank.label))
