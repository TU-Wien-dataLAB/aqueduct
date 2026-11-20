from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from management.models import Request
from management.outcomes import FAILED, Outcome
from management.tests.helpers import make_token

CODES = (None, 0, 200, 302, 399, 400, 404, 428, 429, 430, 499, 500, 503, 599)


class OutcomeTest(TestCase):
    def test_each_status_code_has_the_same_outcome_in_python_and_in_the_database(self):
        token = make_token(get_user_model().objects.create(username="ana"), "key")
        now = timezone.now()
        for code in CODES:
            with self.subTest(code=code):
                request = Request.objects.create(token=token, timestamp=now, status_code=code)
                found = Request.objects.filter(pk=request.pk)
                outcome = Outcome.of(code)
                matched = {o for o in Outcome if found.filter(o.matches).exists()}
                self.assertEqual(matched, set() if outcome is None else {outcome})
                failed = outcome is not None and outcome.failed
                self.assertEqual(found.filter(FAILED).exists(), failed)
