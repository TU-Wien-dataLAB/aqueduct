from management.models import Snippet, SnippetType


def seed_active_config(code: str, name: str = "test-config") -> Snippet:
    Snippet.objects.filter(type=SnippetType.CONFIG, active=True).update(active=False)
    snippet, _ = Snippet.objects.update_or_create(
        name=name, type=SnippetType.CONFIG, defaults={"active": True, "code": code}
    )
    return snippet


SNIPPET_DEFAULT = """\
class MyConfig(ConfigSnippet):
    pass
"""


SNIPPET_ORG_FROM_FIRST_GROUP = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        groups = claims.get("groups") or []
        return groups[0] if groups else None
"""


SNIPPET_ORG_CUSTOM = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        return claims.get("custom_org")
"""


SNIPPET_ORG_LAST_GROUP = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        return (claims.get("groups") or [""])[-1]
"""


SNIPPET_TEAM_NAMES_AND_MAP = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        return "default"

    def team_names(self, claims):
        groups = claims.get("groups") or []
        return [g for g in groups if g.startswith("E")]

    def display_team_names(self, team_names):
        return [(t.split("-", maxsplit=1)[0], t) for t in team_names]
"""


SNIPPET_TEAM_NAMES_KEEP_FULL = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        return "default"

    def team_names(self, claims):
        groups = claims.get("groups") or []
        return [g for g in groups if g.startswith("E")]

    def display_team_names(self, team_names):
        return [(t, t) for t in team_names]
"""


SNIPPET_TEAM_NAMES_PREFIX = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        return ["E123", "E456"]

    def display_team_names(self, team_names):
        return [(n, n) for n in team_names]
"""


SNIPPET_TEAM_NAMES_NON_LIST = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        return "not-a-list"
"""


SNIPPET_TEAM_NAMES_NONE = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        return None
"""


SNIPPET_TEAM_NAMES_RAISES = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        raise RuntimeError("boom")
"""


SNIPPET_DISPLAY_MAP_NONE = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        groups = claims.get("groups") or []
        return [g for g in groups if g.startswith("E")]

    def display_team_names(self, team_names):
        return None
"""


SNIPPET_DISPLAY_MAP_NON_LIST = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        groups = claims.get("groups") or []
        return [g for g in groups if g.startswith("E")]

    def display_team_names(self, team_names):
        return "nope"
"""


SNIPPET_DISPLAY_MAP_BAD_ENTRIES = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        groups = claims.get("groups") or []
        return [g for g in groups if g.startswith("E")]

    def display_team_names(self, team_names):
        return [("E123", "E123-Students"), ("bad",), (None, "x"), 42]
"""


SNIPPET_NO_TEAMS = """\
class MyConfig(ConfigSnippet):
    def team_names(self, claims):
        return []

    def display_team_names(self, team_names):
        return []
"""


SNIPPET_USER_GROUP_BASIC = """\
class MyConfig(ConfigSnippet):
    def user_group(self, claims):
        groups = claims.get("groups") or []
        if "E123-Students" in groups:
            return "admin"
        if "E456-OrgAdmins" in groups:
            return "org-admin"
        return "user"
"""


SNIPPET_USER_GROUP_INVALID = """\
class MyConfig(ConfigSnippet):
    def user_group(self, claims):
        return "superadmin"
"""


SNIPPET_USER_GROUP_RAISES = """\
class MyConfig(ConfigSnippet):
    def user_group(self, claims):
        raise RuntimeError("boom")
"""


def sample_extract_teams(claims) -> list[str]:
    """
    Sample implementation that extracts team names from claims.
    Filters groups starting with 'E'.
    Example: {'groups': ['E123-Students', 'Other']} -> ['E123-Students']
    """
    groups = claims.get("groups") or []
    return [group for group in groups if group.startswith("E")]


def sample_map_team_names(team_names: list[str]) -> list[tuple[str, str]]:
    """
    Sample implementation that maps team names to display team names.
    Extracts team names (removes suffix after dash).
    Example: ['E123-Students'] -> [('E123', 'E123-Students')]
    """
    result = []
    for team_name in team_names:
        display_team_name = team_name.split("-", maxsplit=1)[0]
        result.append((display_team_name, team_name))
    return result


def sample_team_names(claims) -> list[tuple[str, str]]:
    """
    Backward compatibility wrapper. Use sample_extract_teams + sample_map_team_names instead.
    """
    team_names = sample_extract_teams(claims)
    return sample_map_team_names(team_names)


def extract_teams_keep_full(claims) -> list[str]:
    """
    Extract team names, keeping full group name.
    Example: {'groups': ['E123-Students']} -> ['E123-Students']
    """
    groups = claims.get("groups") or []
    return [group for group in groups if group.startswith("E")]


def map_team_names_keep_full(team_names: list[str]) -> list[tuple[str, str]]:
    """
    Map team names keeping full name as display team name.
    Example: ['E123-Students'] -> [('E123-Students', 'E123-Students')]
    """
    return [(group, group) for group in team_names]


def extract_teams_with_prefix(claims) -> list[str]:
    """
    Extract team names starting with 'E'.
    Example: {'groups': ['E123-Students']} -> ['E123-Students']
    """
    groups = claims.get("groups") or []
    return [group for group in groups if group.startswith("E")]


def map_team_names_with_prefix(team_names: list[str]) -> list[tuple[str, str]]:
    """
    Map team names adding 'Team-' prefix.
    Example: ['E123-Students'] -> [('Team-E123-Students', 'E123-Students')]
    """
    return [(f"Team-{group}", group) for group in team_names]


def extract_teams_strip_suffix(claims) -> list[str]:
    """
    Extract team names starting with 'E'.
    Example: {'groups': ['E123-Students']} -> ['E123-Students']
    """
    groups = claims.get("groups") or []
    return [group for group in groups if group.startswith("E")]


def map_team_names_strip_suffix(team_names: list[str]) -> list[tuple[str, str]]:
    """
    Map team names stripping suffix after dash.
    Example: ['E123-Students'] -> [('E123', 'E123-Students')]
    """
    return [(group.split("-", maxsplit=1)[0], group) for group in team_names]


def extract_teams_empty(claims) -> list[str]:
    """
    Extract no team names (causes deletion).
    """
    return []


def map_team_names_empty(team_names: list[str]) -> list[tuple[str, str]]:
    """
    Map to empty list (causes deletion).
    """
    return []


def team_names_keep_full(claims) -> list[tuple[str, str]]:
    return map_team_names_keep_full(extract_teams_keep_full(claims))


def team_names_with_prefix(claims) -> list[tuple[str, str]]:
    return map_team_names_with_prefix(extract_teams_with_prefix(claims))


def team_names_strip_suffix(claims) -> list[tuple[str, str]]:
    return map_team_names_strip_suffix(extract_teams_strip_suffix(claims))


def team_names_empty(claims) -> list[tuple[str, str]]:
    return []
