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
