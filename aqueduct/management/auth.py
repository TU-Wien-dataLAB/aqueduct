import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from mozilla_django_oidc.auth import OIDCAuthenticationBackend

from .models import Org, Team, TeamMembership, UserProfile

User = get_user_model()

log = logging.getLogger("aqueduct")


def default_org_name_from_groups(groups: list[str]) -> str | None:
    """
    Default implementation to extract organization name, which returns the first group in the list.
    Override this or set ORG_NAME_FROM_OIDC_GROUPS_FUNCTION in settings.
    """
    if not groups:
        return None
    return groups[0]


def get_org_name_from_groups(groups) -> str | None:
    """
    Extracts the organization name from the user's groups.
    """
    if hasattr(settings, "ORG_NAME_FROM_OIDC_GROUPS_FUNCTION"):
        return settings.ORG_NAME_FROM_OIDC_GROUPS_FUNCTION(groups)
    return default_org_name_from_groups(groups)


class OIDCBackend(OIDCAuthenticationBackend):
    def _groups(self, claims) -> list[str]:
        return claims.get("groups", settings.OIDC_DEFAULT_GROUPS)

    def _org(self, groups: list[str]) -> Org | None:
        org_name = get_org_name_from_groups(groups)
        if not org_name:
            return None  # Authentication fails if no org can be determined
        org, _created = Org.objects.get_or_create(name=org_name)
        return org

    def _get_teams_from_groups(self, groups: list[str]) -> list[str]:
        """
        Get list of team names to create/join from OAuth groups.
        Calls OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION setting.
        Returns empty list on error or if feature is disabled.
        """
        if not getattr(settings, "ENABLE_OAUTH_GROUP_MANAGEMENT", False):
            return []

        func = getattr(settings, "OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION", lambda groups: [])
        try:
            team_names = func(groups)
            if not isinstance(team_names, list):
                log.error("OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION must return a list")
                return []
            return [name.strip() for name in team_names if name and isinstance(name, str)]
        except Exception as e:
            log.exception("Error calling OAUTH_TEAM_NAMES_FROM_GROUPS_FUNCTION: %s", e)
            return []

    def _sync_teams(self, user: User, profile: UserProfile, groups: list[str]) -> None:
        """
        Synchronize team membership based on OAuth groups.

        - Creates teams if ENABLE_OAUTH_GROUP_CREATION=True and team doesn't exist
        - Adds user to teams via TeamMembership
        - Removes user from teams no longer in their groups
        - Uses transactions for atomicity
        - Logs all changes
        - Respects org boundaries (teams must belong to user's org)
        """
        if not getattr(settings, "ENABLE_OAUTH_GROUP_MANAGEMENT", False):
            return

        team_names = self._get_teams_from_groups(groups)
        if not team_names:
            return

        org = profile.org

        with transaction.atomic():
            existing_memberships = set(
                TeamMembership.objects.filter(user_profile=profile).values_list(
                    "team__name", flat=True
                )
            )

            target_teams = set(team_names)

            teams_to_add = target_teams - existing_memberships
            teams_to_remove = existing_memberships - target_teams

            enable_creation = getattr(settings, "ENABLE_OAUTH_GROUP_CREATION", True)

            for team_name in teams_to_add:
                try:
                    team = Team.objects.get(name=team_name, org=org)
                    log.info("Reused existing team '%s' for org '%s'", team_name, org.name)
                except Team.DoesNotExist:
                    if not enable_creation:
                        log.info(
                            "Skipping team creation for '%s' (ENABLE_OAUTH_GROUP_CREATION=False)",
                            team_name,
                        )
                        continue
                    team = Team.objects.create(name=team_name, org=org, oauth_group_name=team_name)
                    log.info("Created team '%s' for org '%s'", team_name, org.name)

                TeamMembership.objects.get_or_create(user_profile=profile, team=team)
                log.info("Added user '%s' to team '%s' (%s)", user.email, team_name, org.name)

            for team_name in teams_to_remove:
                try:
                    team = Team.objects.get(name=team_name, org=org)
                    TeamMembership.objects.filter(user_profile=profile, team=team).delete()
                    log.info(
                        "Removed user '%s' from team '%s' (%s)", user.email, team_name, org.name
                    )
                except Team.DoesNotExist:
                    log.warning(
                        "Team '%s' not found for removal (org: %s, user: %s)",
                        team_name,
                        org.name,
                        user.email,
                    )

    def create_user(self, claims) -> User | None:
        groups = self._groups(claims)
        org = self._org(groups)
        if not org:
            return None  # Authentication fails if no org can be determined

        user = super().create_user(claims)
        profile = UserProfile.objects.create(user=user, org=org)

        # Check if user is admin
        if hasattr(settings, "ADMIN_GROUP"):
            is_admin = settings.ADMIN_GROUP in groups
        else:
            is_admin = False
        user.is_staff = is_admin
        user.is_superuser = is_admin
        if is_admin:
            profile.group = "admin"
        else:
            profile.group = "user"

        user.save()
        profile.save()

        log.info("Created user '%s' (%s)", user.email, profile.group)

        # Sync team membership from OAuth groups
        self._sync_teams(user, profile, groups)

        return user

    def update_user(self, user, claims) -> User:
        """Update existing user with new claims, if necessary save, and return user"""
        groups = self._groups(claims)
        org = self._org(groups)

        try:
            profile = UserProfile.objects.get(user=user)
            if profile.org != org:
                profile.org = org
                profile.save()
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=user, org=org)

        # Check if user is admin
        if hasattr(settings, "ADMIN_GROUP"):
            is_admin = settings.ADMIN_GROUP in groups
        else:
            is_admin = False

        if is_admin:
            profile.group = "admin"
        elif user.is_superuser:
            # If user was admin make them "user"
            profile.group = "user"

        user.is_staff = is_admin
        user.is_superuser = is_admin
        user.save()
        profile.save()

        log.info("Updated user '%s' (%s)", user.email, profile.group)

        # Sync team membership from OAuth groups
        self._sync_teams(user, profile, groups)

        return user
