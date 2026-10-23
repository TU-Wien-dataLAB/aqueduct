from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.test.utils import CaptureQueriesContext
from openai.types.chat import ChatCompletion

from gateway.tests.utils.base import ChatCompletionsBase
from management.models import Org, Request, ServiceAccount, Team, Token, UserProfile

User = get_user_model()


class TokenRateLimitTest(ChatCompletionsBase):
    def setUp(self):
        super().setUp()
        # Rate-limit counts now live in the cache (LocMemCache under test), which
        # is shared in-process across tests within the same window. Clear it so
        # bucket counts from a previous test in the same minute/hour/day don't
        # bleed into this one (DB transaction rollback no longer resets them).
        cache.clear()

    def tearDown(self):
        cache.clear()
        super().tearDown()

    def _setup_limits(self, kind: str, field: str, value: int):
        """
        Set a rate limit for the given kind ('org', 'team', 'user') and field.
        Returns the object whose limit was set.

        For 'team', also ensure the service account is associated with the test token.
        """
        if kind == "org":
            org = Org.objects.get(name="E060")
            setattr(org, field, value)
            org.save(update_fields=[field])
            return org
        if kind == "team":
            team = Team.objects.get(name="Whale")
            setattr(team, field, value)
            team.save(update_fields=[field])
            # Ensure a service account exists for the team and associate it with the token
            # Only associate the SA with the token if it does not already have a service account

            service_account = ServiceAccount.objects.create(team=team, name="Whale Service")

            token = Token.objects.filter(
                key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN)
            ).first()
            if not token:
                raise RuntimeError("Could not find Token associated with AQUEDUCT_ACCESS_TOKEN.")

            # Only set the service_account if it is not already set
            if getattr(token, "service_account_id", None) is None:
                token.service_account = service_account
                token.save(update_fields=["service_account"])
            elif token.service_account_id != service_account.id:
                # If the token is already associated with a different SA, raise an error
                raise RuntimeError(
                    f"Token is already associated with a different service account "
                    f"(id={token.service_account_id})."
                )
            # Otherwise, already associated with the correct service account, do nothing
            return team
        if kind == "user":
            user = User.objects.get(username="Me")
            profile = (
                user.profile if hasattr(user, "profile") else UserProfile.objects.get(user=user)
            )
            setattr(profile, field, value)
            profile.save(update_fields=[field])
            return profile
        raise ValueError(f"Unknown kind: {kind}")

    def _rate_limit_test_template(
        self, kind: str, field: str, value: int, messages, max_completion_tokens, limit_desc
    ):
        """
        Generic template for rate limit tests for org, team, or user.
        Uses the Django test client to POST to the chat completion endpoint.
        """
        # Set the limit
        self._setup_limits(kind, field, value)

        response1 = self._send_chat_completion(
            messages, max_completion_tokens=max_completion_tokens
        )
        self.assertEqual(
            response1.status_code,
            200,
            f"Expected 200 OK, got {response1.status_code}: {response1.content}",
        )
        response_json = response1.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertGreater(len(chat_completion.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests),
            1,
            f"There should be exactly one request after first chat completion ({limit_desc}).",
        )
        req = requests[0]
        self.assertIn(self.url, req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        response2 = self._send_chat_completion(
            messages, max_completion_tokens=max_completion_tokens
        )
        self.assertEqual(
            response2.status_code,
            429,
            f"Expected 429 Too Many Requests, got {response2.status_code}: {response2.content}",
        )
        # The exceeded minute bucket ends within a minute
        retry_after = int(response2.headers["Retry-After"])
        self.assertGreaterEqual(retry_after, 1)
        self.assertLessEqual(retry_after, 60)
        # Optionally, check error message
        try:
            error_json = response2.json()
            self.assertTrue(
                "rate limit" in str(error_json).lower() or "429" in str(error_json),
                f"Expected rate limit error message, got: {error_json}",
            )
        except Exception:
            # If not JSON, just check content
            self.assertTrue(
                "rate limit" in response2.content.decode().lower()
                or "429" in response2.content.decode(),
                f"Expected rate limit error message, got: {response2.content}",
            )

    def test_org_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Org 'E060' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_completion_tokens=5,
            limit_desc="org requests_per_minute",
        )

    def test_team_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Team 'Whale' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_completion_tokens=5,
            limit_desc="team requests_per_minute",
        )

    def test_user_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of UserProfile for user 'Me' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_completion_tokens=5,
            limit_desc="user requests_per_minute",
        )

    def test_org_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=1,
            limit_desc="org input_tokens_per_minute",
        )

    def test_team_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=1,
            limit_desc="team input_tokens_per_minute",
        )

    def test_user_rate_limit_input_tokens_per_minute(self):
        """
        Edits input_tokens_per_minute of UserProfile for user 'Me' to 5,
        then makes two requests. The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=1,
            limit_desc="user input_tokens_per_minute",
        )

    def test_org_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=10,
            limit_desc="org output_tokens_per_minute",
        )

    def test_team_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=10,
            limit_desc="team output_tokens_per_minute",
        )

    def test_user_rate_limit_output_tokens_per_minute(self):
        """
        Edits output_tokens_per_minute of UserProfile for user 'Me' to 5,
        then makes two requests. The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_completion_tokens=10,
            limit_desc="user output_tokens_per_minute",
        )

    def test_rate_limit_message_lists_each_exceeded_limit(self):
        self._setup_limits("org", "requests_per_minute", 1)
        self._setup_limits("org", "output_tokens_per_minute", 1)
        first = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(first.status_code, 200)

        blocked = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(blocked.status_code, 429)
        retry_after = blocked.headers["Retry-After"]
        self.assertEqual(
            blocked.json()["error"]["message"],
            f"Rate limit exceeded. Request limit (1/min) (Retry after {retry_after}s), "
            f"Output token limit (1/min) (Retry after {retry_after}s).",
        )

    @patch("gateway.rate_limiting.get_model_request_limit_multiplier", return_value=2.0)
    def test_per_model_request_limit_multiplier_budget(self, mock_multipliers):
        """
        Tests that per-model multipliers act as a budget cost.
        With multiplier 2.0 ("2x Limits") and limit 3:
        Each request costs 1/2.0 = 0.5 budget.
        - Requests 1-6: weighted cost 0, 0.5, 1.0, 1.5, 2.0, 2.5 all < 3, allowed
        - 7th request: weighted = 3.0 >= 3, blocked
        """
        # Set request limit to 3
        self._setup_limits("org", "requests_per_minute", 3)

        # Requests 1-6 should succeed (weighted cost up to 2.5 < 3)
        for i in range(6):
            response = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 OK for request {i + 1}, got {response.status_code}",
            )

        # 7th request should fail (weighted cost = 3.0 >= 3)
        response7 = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(
            response7.status_code,
            429,
            f"Expected 429 Too Many Requests, got {response7.status_code}: {response7.content}",
        )
        retry_after = response7.headers["Retry-After"]
        self.assertEqual(
            response7.json()["error"]["message"],
            f"Rate limit exceeded. Request limit (6/min) (Retry after {retry_after}s).",
        )

    @patch("gateway.rate_limiting.get_model_request_limit_multiplier", return_value=0.5)
    def test_per_model_expensive_multiplier_limits_requests(self, mock_multipliers):
        """
        Tests that a model with multiplier < 1 costs more budget per request.
        With multiplier 0.5 and limit 3:
        Each request costs 1/0.5 = 2.0 budget.
        - 1st request: weighted = 0 < 3, allowed
        - 2nd request: weighted = 2.0 < 3, allowed
        - 3rd request: weighted = 4.0 >= 3, blocked
        """
        # Set request limit to 3
        self._setup_limits("org", "requests_per_minute", 3)

        # First request: weighted = 0 < 3
        response1 = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(response1.status_code, 200)

        # Second request: weighted = 2.0 < 3
        response2 = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(response2.status_code, 200)

        # Third request: weighted = 4.0 >= 3
        response3 = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(response3.status_code, 429)
        retry_after = response3.headers["Retry-After"]
        self.assertEqual(
            response3.json()["error"]["message"],
            f"Rate limit exceeded. Request limit (1.5/min) (Retry after {retry_after}s).",
        )

    def test_check_limits_uses_no_request_table_sql(self):
        """A rate-limited (429) request issues no SQL against ``management_request``.

        ``check_limits`` now reads from the cache, not the ``Request`` table. A
        blocked request returns 429 before ``log_request`` runs, so it must not
        touch ``management_request`` at all (no aggregate SELECT, no INSERT).
        """
        self._setup_limits("org", "requests_per_minute", 1)
        # First request consumes the minute bucket (and creates its Request row).
        first = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(first.status_code, 200)

        # Capture SQL during the second (blocked) request.
        with CaptureQueriesContext(connection) as ctx:
            blocked = self._send_chat_completion(self.MESSAGES, max_completion_tokens=5)
        self.assertEqual(blocked.status_code, 429)
        request_table_queries = [
            q["sql"] for q in ctx.captured_queries if "management_request" in q["sql"]
        ]
        self.assertEqual(
            request_table_queries,
            [],
            "check_limits must not query the Request table; found: "
            + " | ".join(request_table_queries),
        )
