import logging
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from mozilla_django_oidc.auth import OIDCAuthenticationBackend

from management.models import Org, Team, TeamMembership, UserGroup, UserProfile
from management.snippets import get_config_snippet

User = get_user_model()

log = logging.getLogger("aqueduct")


class OIDCBackend(OIDCAuthenticationBackend):
    def _org(self, claims: dict[str, Any]) -> Org | None:
        if not (org_name := get_config_snippet().org_name(claims)):
            return None
        org, _ = Org.objects.get_or_create(name=org_name)
        return org

    def _get_team_names(self, claims: dict[str, Any]) -> list[str]:
        """
        Extract the raw team (group) names from OAuth claims.

        Returns:
            List of team names, or empty list if none/extraction fails.
        """
        if not getattr(settings, "ENABLE_OAUTH_GROUP_MANAGEMENT", False):
            return []

        try:
            team_names = get_config_snippet().team_names(claims)
        except Exception as e:
            log.exception("Error calling config snippet team_names(): %s", e)
            return []

        if not isinstance(team_names, list):
            log.error(
                "Config snippet team_names() must return a list of strings; got %r",
                type(team_names).__name__,
            )
            return []

        return team_names

    def _get_teams(self, team_names: list[str]) -> list[tuple[str, str]]:
        """
        Map raw team names to (display_team_name, team_name) tuples.

        Returns:
            List of (display_team_name, team_name) tuples, or empty list if none.
        """
        try:
            result = get_config_snippet().display_team_names(team_names)
        except Exception as e:
            log.exception("Error calling config snippet display_team_names(): %s", e)
            return []

        if not isinstance(result, list):
            log.error(
                "Config snippet display_team_names() must return a list of tuples; got %r",
                type(result).__name__,
            )
            return []

        team_mappings = []
        for item in result:
            if not isinstance(item, tuple) or len(item) != 2:  # noqa: PLR2004
                continue
            display_team_name, team_name = item
            if not (display_team_name and isinstance(display_team_name, str) and team_name):
                continue
            team_mappings.append((display_team_name.strip(), team_name.strip()))

        return team_mappings

    def _update_user_role(
        self, user: User, profile: UserProfile, team_names: list[str], claims: dict[str, Any]
    ) -> None:
        snippet = get_config_snippet()

        try:
            group = snippet.user_group(claims)
        except Exception as e:
            log.exception("Error calling config snippet user_group(): %s", e)
            group = UserGroup.USER

        if group not in UserGroup.values:
            log.error(
                "Config snippet user_group() returned invalid group %r; falling back to 'user'",
                group,
            )
            group = UserGroup.USER

        admin_group = getattr(settings, "ADMIN_GROUP", "")
        if admin_group and admin_group in team_names:
            group = UserGroup.ADMIN

        admin_superuser_emails = getattr(settings, "ADMIN_SUPERUSER_EMAILS", [])
        is_superuser_email = user.email and user.email.lower() in admin_superuser_emails

        profile.group = group
        profile.save()

        is_superuser = (group == UserGroup.ADMIN) or is_superuser_email
        user.is_staff = is_superuser
        user.is_superuser = is_superuser
        user.save()

    @staticmethod
    def _add_team_membership(profile: UserProfile, teams):
        if not getattr(settings, "ENABLE_OAUTH_GROUP_CREATION", True):
            log.info(
                "Skipping creation of teams %s (ENABLE_OAUTH_GROUP_CREATION=False)",
                sorted(team_name for team_name, _ in teams),
            )
            return

        org = profile.org
        for team_name, original_group_name in teams:
            # Look up by oauth_group_name first, so renaming the mapping
            # function renames existing teams instead of creating duplicates.
            team = Team.objects.filter(oauth_group_name=original_group_name, org=org).first()
            created = False

            if not team:
                team, created = Team.objects.get_or_create(
                    name=team_name, org=org, defaults={"oauth_group_name": original_group_name}
                )

            if team and team.name != team_name:
                collision = (
                    Team.objects.filter(name=team_name, org=org).exclude(pk=team.pk).exists()
                )

                if collision:
                    log.warning(
                        "Cannot rename team '%s' -> '%s': name collision (org: %s, "
                        "oauth_group: '%s'). Reusing existing team as-is.",
                        team.name,
                        team_name,
                        org.name,
                        original_group_name,
                    )
                else:
                    log.info(
                        "Renaming team '%s' -> '%s' (org: %s, oauth_group: '%s')",
                        team.name,
                        team_name,
                        org.name,
                        original_group_name,
                    )
                    team.name = team_name
                    team.save(update_fields=["name"])

            if created:
                log.info("Created team '%s' for org '%s'", team_name, org.name)
            else:
                log.info("Reused existing team '%s' for org '%s'", team_name, org.name)

            TeamMembership.objects.get_or_create(user_profile=profile, team=team)
            log.info("Added user '%s' to team '%s' (%s)", profile.user.email, team_name, org.name)

    @staticmethod
    def _remove_team_membership(profile: UserProfile, teams):
        if not getattr(settings, "ENABLE_OAUTH_GROUP_REMOVAL", True):
            log.info(
                "Skipping removal from teams %s for user '%s' (ENABLE_OAUTH_GROUP_REMOVAL=False)",
                sorted(teams),
                profile.user.email,
            )
            return

        org = profile.org
        for team_name in teams:
            try:
                team = Team.objects.get(name=team_name, org=org)
                TeamMembership.objects.filter(user_profile=profile, team=team).delete()
                log.info(
                    "Removed user '%s' from team '%s' (%s)", profile.user.email, team_name, org.name
                )
            except Team.DoesNotExist:
                log.warning(
                    "Team '%s' not found for removal (org: %s, user: %s)",
                    team_name,
                    org.name,
                    profile.user.email,
                )

    def _sync_team_membership(self, user: User, profile: UserProfile, team_names: list[str]):
        """
        Synchronize team membership based on OAuth claims.

        - Creates teams if ENABLE_OAUTH_GROUP_CREATION=True and team doesn't exist
        - Adds user to teams via TeamMembership
        - Removes user from teams no longer in their groups
        - Uses transactions for atomicity
        - Logs all changes
        - Respects org boundaries (teams must belong to user's org)
        """
        if not getattr(settings, "ENABLE_OAUTH_GROUP_MANAGEMENT", False):
            log.info(
                "Skipping synchronization of teams %s (ENABLE_OAUTH_GROUP_MANAGEMENT=False)",
                sorted(team_name for team_name, _ in team_names),
            )
            return

        if not team_names:
            return

        team_mappings = self._get_teams(team_names=team_names)

        with transaction.atomic():
            existing_memberships = set(
                TeamMembership.objects.filter(user_profile=profile)
                .exclude(team__oauth_group_name__in=["", None])
                .values_list("team__name", flat=True)
            )

            target_team_names = {team_name for team_name, _ in team_mappings}
            team_name_to_original = dict(team_mappings)

            teams_to_add = [
                (name, team_name_to_original[name])
                for name in target_team_names - existing_memberships
            ]
            self._add_team_membership(profile=profile, teams=teams_to_add)

            teams_to_remove = existing_memberships - target_team_names
            self._remove_team_membership(profile=profile, teams=teams_to_remove)

    def create_user(self, claims: dict[str, Any]) -> User | None:
        org = self._org(claims)
        if not org:
            return None  # Authentication fails if no org can be determined

        user = super().create_user(claims)
        profile = UserProfile.objects.create(user=user, org=org)

        team_names = self._get_team_names(claims)

        self._update_user_role(user, profile, team_names, claims)

        log.info("Created user '%s' (%s)", user.email, profile.group)

        self._sync_team_membership(user, profile, team_names)

        return user

    def update_user(self, user, claims: dict[str, Any]) -> User:
        """Update existing user with new claims, if necessary save, and return user"""
        org = self._org(claims)
        if not org:
            return user  # Authentication fails if no org can be determined

        profile, _ = UserProfile.objects.update_or_create(user=user, defaults={"org": org})

        team_names = self._get_team_names(claims)

        self._update_user_role(user, profile, team_names, claims)

        log.info("Updated user '%s' (%s)", user.email, profile.group)

        self._sync_team_membership(user, profile, team_names)

        return user
