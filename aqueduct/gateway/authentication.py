import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import BaseBackend
from django.utils import timezone

from management.models import Token

logger = logging.getLogger("aqueduct")
User = get_user_model()


def token_from_request(request) -> str | None:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        # No valid header, authentication cannot proceed with this backend.
        logger.debug("TokenAuthenticationBackend: No 'Bearer ' auth header found.")
        return None

    try:
        token_key = auth_header.split(" ")[1]
        if not token_key:
            raise IndexError
    except IndexError:
        logger.warning("Authentication failed: Empty or badly formatted Bearer token in header.")
        return None
    else:
        logger.debug(
            "TokenAuthenticationBackend: Attempting to authenticate with token starting with %s",
            Token._generate_preview(token_key),
        )
        return token_key


class TokenAuthenticationBackend(BaseBackend):
    """
    Authenticates against a Bearer token provided in the Authorization header,
    using the aqueduct.management.models.Token model.
    """

    def authenticate(self, request, **kwargs) -> User | None:
        """
        Authenticates the request based on the 'Authorization: Bearer <token>' header.

        Args:
            request: The HttpRequest object.
            **kwargs: Allows for Django's internal flexibility but not used directly here.

        Returns:
            The User object associated with the token if authentication is successful.
            None otherwise.
        """
        logger.debug("TokenAuthenticationBackend: authenticate method called.")
        token_key = token_from_request(request)
        if not token_key:
            return None

        # Find the token using the provided key
        token_instance = Token.find_by_key(token_key)

        if not token_instance:
            logger.warning(
                "Authentication failed: Invalid token provided (starts with %s). Token.find_by_key returned None.",
                Token._generate_preview(token_key),
            )
            return None

        # Optional: Check for token expiry
        if token_instance.expires_at and token_instance.expires_at < timezone.now():
            logger.warning(
                "Authentication failed: Token %s (%s) has expired.", token_instance.name, token_instance.key_preview
            )
            return None

        # Token is valid, return the associated user
        # Ensure the related user exists (should always be true due to ForeignKey constraints)
        if not token_instance.user:
            logger.error("Critical: Valid token %s found but has no associated user.", token_instance.id)
            return None

        logger.info(
            "Authentication successful via token: %s (%s) for user %s",
            token_instance.name,
            token_instance.key_preview,
            token_instance.user.email,
        )
        # Attach the token to the user object for potential use in views/middleware downstream
        # Note: Modifying request.user directly isn't standard, attaching to request is better if needed
        # For now, we just return the user as required by authenticate()
        # If the token object itself is needed later, consider attaching it to request
        # E.g. in middleware: request.auth_token = token_instance
        return token_instance.user

    def get_user(self, user_id) -> User | None:
        """
        Retrieves a user instance given the user_id (primary key).
        Required by Django's authentication framework.
        """
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
