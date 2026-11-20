from dataclasses import replace
from datetime import timedelta
from unittest import mock

from django.urls import reverse

from management.tests.helpers import EMBEDDING, UsagePageTestBase, make_token
from management.usage_report import TOP_KEYS

ALL_KEYS_LINK = f'class="link-plain text-sm" href="{reverse("usage_keys")}'


class KeyUsagePageTestBase(UsagePageTestBase):
    def keys_page(self, user, **params):
        self.client.force_login(user)
        return self.client.get(reverse("usage_keys"), params)

    def names(self, user, **params) -> list[str]:
        response = self.keys_page(user, **params)
        self.assertEqual(response.status_code, 200)
        return [key.name for key in response.context["keys"]]


class KeyUsageAccessTest(KeyUsagePageTestBase):
    def test_user_with_several_keys_sees_only_their_own(self):
        make_token(self.solo, "solo-desktop")
        response = self.keys_page(self.solo)
        self.assert_keys(response, "solo-laptop")
        self.assertNotContains(response, '<span class="sr-only">Key: </span>')

    def test_org_admin_sees_every_key_of_their_org(self):
        self.assert_keys(self.keys_page(self.org_admin), "solo-laptop", "member-laptop", "pipeline")

    def test_a_key_parameter_does_not_narrow_or_widen_the_list(self):
        self.assertEqual(
            self.names(self.member, key=self.foreign_key.id), ["pipeline", "member-laptop"]
        )
        self.assertEqual(
            self.names(self.org_admin, key=self.solo_key.id),
            ["pipeline", "member-laptop", "solo-laptop"],
        )


class KeyUsageSearchTest(KeyUsagePageTestBase):
    def test_search_matches_key_names_within_what_the_viewer_sees(self):
        self.assertEqual(self.names(self.member, q="LAPTOP"), ["member-laptop"])
        self.assertEqual(self.names(self.member, q="LAP\x00TOP"), ["member-laptop"])
        self.assertEqual(self.names(self.org_admin, q="laptop"), ["member-laptop", "solo-laptop"])
        self.assertEqual(self.names(self.other_org_admin, q="solo"), [])

    def test_only_a_viewer_who_sees_the_whole_org_searches_by_owner_email(self):
        self.assertEqual(self.names(self.member, q="team-admin@"), [])
        self.assertEqual(self.names(self.org_admin, q="team-admin@"), ["pipeline"])
        self.assertEqual(self.names(self.org_admin, q="foreigner@"), [])

    def test_search_counts_the_matching_keys_and_keeps_the_period_share(self):
        response = self.keys_page(self.org_admin, q="pipeline")
        self.assertEqual(response.context["count"], 1)
        self.assertContains(response, "1 key")
        (pipeline,) = response.context["keys"]
        self.assertEqual((pipeline.share, pipeline.width), ("53%", 100))

    def test_the_key_pickers_search_lists_no_keys_here(self):
        response = self.keys_page(self.org_admin, key_q="laptop")
        self.assertNotContains(response, "Matching keys")
        self.assertNotContains(response, f"key={self.solo_key.id}")

    def test_search_without_a_match_says_so(self):
        self.assertContains(self.keys_page(self.member, q="nothing"), "No key that matches")


class KeyUsageOrderTest(KeyUsagePageTestBase):
    def setUp(self):
        super().setUp()
        for hours in (2, 3):
            self.log(self.solo_key, timedelta(hours=hours), 200, input_tokens=10)

    def test_keys_sort_by_tokens_by_default(self):
        self.assertEqual(self.names(self.org_admin), ["pipeline", "member-laptop", "solo-laptop"])

    def test_keys_sort_by_requests_or_name(self):
        self.assertEqual(
            self.names(self.org_admin, sort="requests"),
            ["solo-laptop", "pipeline", "member-laptop"],
        )
        self.assertEqual(
            self.names(self.org_admin, sort="name"), ["member-laptop", "pipeline", "solo-laptop"]
        )
        self.assertEqual(
            self.names(self.org_admin, sort="bogus"), ["pipeline", "member-laptop", "solo-laptop"]
        )

    def test_filters_narrow_the_list(self):
        self.log(self.member_key, timedelta(days=10), 200, input_tokens=99_000)
        self.log(self.member_key, timedelta(hours=1), 200, model=EMBEDDING, input_tokens=5)
        self.assertEqual(self.names(self.org_admin, span="1m")[0], "member-laptop")
        self.assertEqual(self.names(self.org_admin, model=EMBEDDING), ["member-laptop"])
        self.assertEqual(self.names(self.org_admin, span="1h"), [])


@mock.patch("management.views.usage.PAGE_SIZE", 2)
class KeyUsagePagingTest(KeyUsagePageTestBase):
    def setUp(self):
        super().setUp()
        for n in range(3):
            key = make_token(self.solo, f"spare-{n}")
            self.log(key, timedelta(hours=1), 200, input_tokens=10 - n)

    def test_pages_hold_page_size_keys_in_order(self):
        self.assertEqual(self.names(self.solo), ["solo-laptop", "spare-0"])
        self.assertEqual(self.names(self.solo, page=2), ["spare-1", "spare-2"])
        response = self.keys_page(self.solo, page=2)
        self.assertEqual(response.context["count"], 4)
        self.assertContains(response, "3\u20134 of 4")

    def test_a_page_past_the_end_or_unreadable_shows_the_nearest_page(self):
        self.assertEqual(self.names(self.solo, page=99), ["spare-1", "spare-2"])
        self.assertEqual(self.names(self.solo, page="x"), ["solo-laptop", "spare-0"])
        self.assertEqual(self.names(self.solo, page="0"), ["solo-laptop", "spare-0"])

    def test_bars_scale_to_the_busiest_key_on_every_page(self):
        widths = [key.width for key in self.keys_page(self.solo, page=2).context["keys"]]
        self.assertEqual(widths, [1, 1])

    def test_paging_keeps_the_filters_and_a_filter_change_goes_back_to_page_one(self):
        response = self.keys_page(self.solo, page=1, span="1m", sort="name", q="spare")
        self.assertContains(response, 'href="?page=2&amp;span=1m&amp;sort=name&amp;q=spare"')
        response = self.keys_page(self.solo, page=2, span="1m")
        self.assertContains(response, 'href="?span=1d"')


class KeyUsageMatchesOverviewTest(KeyUsagePageTestBase):
    def test_first_rows_are_the_overviews_top_keys(self):
        for n in range(12):
            key = make_token(self.org_admin, f"busy-{n}")
            self.log(key, timedelta(hours=1), 200, input_tokens=100 * (n + 1), output_tokens=n)
        for params in ({}, {"span": "1m"}, {"model": EMBEDDING}, {"scope": "mine"}):
            top = self.page(self.org_admin, **params).context["keys"]
            listed = self.keys_page(self.org_admin, **params).context["keys"]
            ranks = [replace(share, limits=None) for share in listed[: len(top)]]
            self.assertEqual(ranks, [replace(share, limits=None) for share in top], params)

    def busy_keys(self, user, n: int) -> None:
        for i in range(n):
            key = make_token(user, f"busy-{user.username}-{i}")
            self.log(key, timedelta(hours=1), 200, model=EMBEDDING, input_tokens=10)

    def test_overview_links_to_all_keys_with_its_filters_when_top_keys_leaves_keys_out(self):
        self.busy_keys(self.org_admin, TOP_KEYS + 1)
        response = self.page(self.org_admin, span="1m", model=EMBEDDING, scope="mine")
        url = f"{reverse('usage_keys')}?scope=mine&amp;model={EMBEDDING}&amp;span=1m"
        self.assertContains(response, f'class="link-plain text-sm" href="{url}">All keys')

    def test_all_keys_link_shows_only_while_top_keys_leaves_keys_out(self):
        whole_org = ({}, {"span": "1d"})
        self.busy_keys(self.org_admin, TOP_KEYS - 3)
        for params in (*whole_org, {"scope": "mine"}):
            self.assertNotContains(self.page(self.org_admin, **params), ALL_KEYS_LINK)
        self.busy_keys(self.solo, 1)
        for params in whole_org:
            self.assertContains(self.page(self.org_admin, **params), ALL_KEYS_LINK)
        self.assertNotContains(self.page(self.org_admin, scope="mine"), ALL_KEYS_LINK)

    def test_breadcrumb_returns_to_usage_with_the_filters(self):
        response = self.keys_page(self.org_admin, scope="mine", span="1d", q="lap")
        self.assertContains(response, f'href="{reverse("usage")}?scope=mine&amp;span=1d">Usage')
        self.assertNotContains(response, 'aria-label="Usage"')


class KeyUsageHeadingsTest(KeyUsagePageTestBase):
    def test_column_headings_sort_the_list(self):
        response = self.keys_page(self.org_admin, span="1w")
        self.assertContains(response, "3 keys</span>")
        self.assertNotContains(response, 'aria-label="Sort by"')
        html = response.content.decode()
        self.assertIn('aria-sort="descending"><a class="inline-flex', html)
        self.assertIn('href="?span=1w&amp;sort=name"', html)
        self.assertIn('href="?span=1w&amp;sort=requests"', html)
        response = self.keys_page(self.org_admin, span="1w", sort="name", page="2")
        self.assertContains(response, 'aria-sort="ascending"')
        self.assertContains(response, 'href="?span=1w"')
