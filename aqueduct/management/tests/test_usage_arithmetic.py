from django.test import SimpleTestCase

from management.models import Request, Usage


class UsageArithmeticTest(SimpleTestCase):
    def test_usage_arithmetic_preserves_token_subsets(self):
        first = Usage(
            input_tokens=100, output_tokens=20, cached_input_tokens=80, reasoning_tokens=12
        )
        second = Usage(
            input_tokens=50, output_tokens=10, cached_input_tokens=30, reasoning_tokens=6
        )
        combined = first + second
        self.assertEqual(
            combined,
            Usage(input_tokens=150, output_tokens=30, cached_input_tokens=110, reasoning_tokens=18),
        )
        self.assertEqual(combined.total_tokens, 180)
        self.assertEqual(combined - second, first)
