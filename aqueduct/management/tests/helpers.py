"""
Shared test helpers for management app tests.
"""


def sample_team_names(claims) -> list[tuple[str, str]]:
    """
    Sample implementation that filters groups starting with 'E'
    and extracts team names (removes suffix after dash).
    Returns list of (team_name, original_group_name) tuples.
    Example: {'groups': ['E123-Students']} -> [('E123', 'E123-Students')]
    """
    groups = claims.get("groups") or []
    result = []
    for group in groups:
        if group.startswith("E"):
            team_name = group.split("-", maxsplit=1)[0]
            result.append((team_name, group))
    return result


def team_names_keep_full(claims) -> list[tuple[str, str]]:
    """
    Helper that keeps full group name as team name.
    Example: {'groups': ['E123-Students']} -> [('E123-Students', 'E123-Students')]
    """
    groups = claims.get("groups") or []
    result = []
    for group in groups:
        if group.startswith("E"):
            result.append((group, group))
    return result


def team_names_with_prefix(claims) -> list[tuple[str, str]]:
    """
    Sample function that adds 'Team-' prefix to group names.
    Example: {'groups': ['E123-Students']} -> [('Team-E123-Students', 'E123-Students')]
    """
    groups = claims.get("groups") or []
    result = []
    for group in groups:
        if group.startswith("E"):
            result.append((f"Team-{group}", group))
    return result


def team_names_strip_suffix(claims) -> list[tuple[str, str]]:
    """
    Sample function that strips suffix after dash.
    Example: {'groups': ['E123-Students']} -> [('E123', 'E123-Students')]
    """
    groups = claims.get("groups") or []
    result = []
    for group in groups:
        if group.startswith("E"):
            result.append((group.split("-", maxsplit=1)[0], group))
    return result


def team_names_empty(claims) -> list[tuple[str, str]]:
    """
    Sample function that returns empty list (causes deletion).
    """
    return []


def custom_filter_team_names(claims) -> list[tuple[str, str]]:
    """
    Custom filter that only allows specific group names.
    Example: {'groups': ['E123', 'E456', 'E789']} -> [('E123', 'E123'), ('E456', 'E456')]
    """
    groups = claims.get("groups") or []
    result = []
    for group in groups:
        if group in {"E123", "E456"}:
            result.append((group, group))
    return result
