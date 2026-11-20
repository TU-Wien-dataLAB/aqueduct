from django.test import SimpleTestCase

from management.formatting import compact, decimal, duration, grouped, percent


class CompactNumbersTest(SimpleTestCase):
    def test_shortens_with_one_decimal_and_carries_to_the_next_unit(self):
        cases = {
            0: "0",
            999: "999",
            1_000: "1K",
            1_049: "1K",
            1_050: "1.1K",
            1_150: "1.2K",
            1_450: "1.5K",
            131_072: "131.1K",
            999_949: "999.9K",
            999_950: "1M",
            29_331_900_000: "29.3B",
            1_500_000_000: "1.5B",
            999_950_000_000: "1T",
            2_300_000_000_000: "2.3T",
            4_000_000_000_000_000: "4000T",
            -1_500: "-1.5K",
        }
        for count, expected in cases.items():
            with self.subTest(count=count):
                self.assertEqual(compact(count), expected)


class NumberFormatsTest(SimpleTestCase):
    def test_numbers_are_grouped_and_durations_pick_their_unit(self):
        cases = {
            "grouped": (grouped(1_440), "1,440"),
            "percent": (percent(12, 100), "12%"),
            "no percent of nothing": (percent(1, 0), "—"),
            "half a percent rounds up": (percent(1, 200), "1%"),
            "seconds": (duration(5_640), "5.6 s"),
            "half a tenth of a second rounds up": (duration(1_250), "1.3 s"),
            "no negative zero": (decimal(-0.04, 1), "0.0"),
            "milliseconds": (duration(120), "120 ms"),
            "unknown": (duration(None), "n/a"),
        }
        for name, (shown, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(shown, expected)
