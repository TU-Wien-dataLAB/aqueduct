class OAuthConfigurationError(Exception):
    """Raised when OAuth configuration is invalid or misconfigured."""


class OAuthFunctionError(Exception):
    """Raised when a configured OAuth function (e.g., OAUTH_TEAM_NAMES_FUNCTION)
    returns an invalid result or raises an exception."""
