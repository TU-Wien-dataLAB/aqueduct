from datetime import timedelta

from django.contrib.auth import get_user_model
from django.urls import reverse

from management.models import Request, TeamMembership
from management.outcomes import Outcome
from management.tests.helpers import CHAT, EMBEDDING, RequestsPageTestBase, join, listed, make_token

User = get_user_model()


class RequestsScopeTest(RequestsPageTestBase):

    def test_log_opens_on_24_hours_and_an_unknown_time_frame_falls_back_to_it(self):
        self.log(self.solo_key, "/yesterday", ago=timedelta(hours=30))
        for params in ({}, {"span": "all"}, {"span": "bogus"}):
            response = self.get(self.solo, **params)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.paths_on(response), {"/own"})
            self.assertNotContains(response, ">/yesterday</div>")
            self.assertContains(response, 'aria-current="page">24 hours</a>')

    def test_a_30_day_link_still_opens_30_days(self):
        self.log(self.solo_key, "/month-old", ago=timedelta(days=29))
        self.log(self.solo_key, "/overdue", ago=timedelta(days=32))
        response = self.get(self.solo, span="1m")
        self.assertContains(response, ">/month-old</div>")
        self.assertNotContains(response, ">/overdue</div>")
        self.assertContains(response, 'aria-current="page">30 days</a>')

    def test_a_key_pages_link_to_an_older_request_opens_it(self):
        older = self.log(
            self.solo_key, "/last-week", ago=timedelta(days=5), user_agent="old-agent/1"
        )
        self.client.force_login(self.solo)
        page = self.client.get(reverse("token_detail", args=[self.solo_key.id])).content.decode()
        link = f"{reverse('requests')}?key={self.solo_key.id}&amp;span=1m&amp;request={older.id}"
        self.assertIn(link, page)
        self.assertContains(self.client.get(link.replace("&amp;", "&")), "old-agent/1")

    def test_empty_log_says_whether_filters_hide_requests(self):
        self.assertContains(
            self.get(self.solo, model=EMBEDDING), "No request matches these filters."
        )
        self.assertContains(
            self.get(self.foreigner, key=self.foreign_key.id, status="5xx"),
            "No request matches these filters.",
        )
        Request.objects.all().delete()
        self.assertContains(self.get(self.solo), "No requests in the last 24 hours.")

    def test_time_frame_limits_the_log(self):
        self.log(self.solo_key, "/last-week", ago=timedelta(days=3))
        self.assertContains(self.get(self.solo, span="1w"), ">/last-week</div>")
        recent = self.get(self.solo)
        self.assertNotContains(recent, ">/last-week</div>")
        self.assertEqual(self.paths_on(recent), {"/own"})

    def test_page_shows_no_ip_address(self):
        self.log(self.solo_key, "/with-ip", ip_address="203.0.113.7", user_agent="probe/1")
        response = self.get(self.solo, request=Request.objects.get(path="/with-ip").id)
        self.assertContains(response, "probe/1")
        self.assertNotContains(response, "203.0.113.7")


class RequestsFilterTest(RequestsPageTestBase):
    def test_status_tabs_count_and_filter(self):
        self.log(self.solo_key, "/limited", status_code=429)
        self.log(self.solo_key, "/invalid", status_code=400)
        self.log(self.solo_key, "/broken", status_code=502)
        response = self.get(self.solo, status="429")
        tabs = {tab.value: tab.count for tab in response.context["log"].tabs}
        self.assertEqual(tabs, {None: 4, "failed": 2, "2xx": 1, "429": 1, "4xx": 1, "5xx": 1})
        content = response.content.decode()
        for outcome in Outcome:
            self.assertIn(f'{outcome.label} <span class="text-xs font-normal opacity-80">', content)
            self.assertIn(f">{outcome.codes}</span>", content)
        self.assertIn(">/limited</div>", content)
        self.assertNotIn(">/own</div>", content)
        self.assertNotIn(">/invalid</div>", content)

    def test_each_tab_counts_the_rows_it_lists_for_every_role(self):
        self.log(self.solo_key, "/limited", status_code=429)
        self.log(self.others_key, "/invalid", status_code=400)
        self.log(self.team_key, "/broken", status_code=502)
        self.log(self.side_key, "/unfinished", status_code=None)
        self.log(self.foreign_key, "/foreign-broken", status_code=500)
        for user in (self.solo, self.member, self.org_admin, self.admin, self.other_org_admin):
            tabs = {tab.value: tab.count for tab in self.get(user).context["log"].tabs}
            self.assertEqual(set(tabs), {None, "failed", "2xx", "429", "4xx", "5xx"})
            for value, count in tabs.items():
                with self.subTest(user=user.username, tab=value):
                    response = self.get(user, **({} if value is None else {"status": value}))
                    self.assertEqual(len(listed(response)), count)
                    self.assertEqual(response.context["log"].count, count)

    def test_model_list_of_a_user_is_what_their_keys_may_use_or_used(self):
        self.org.excluded_models = [CHAT, EMBEDDING, "dall-e-2"]
        self.org.save()
        self.team.excluded_models = ["dall-e-2"]
        self.team.merge_exclusion_lists = False
        self.team.save()
        owner = self.get(self.solo).context["filters"].models
        self.assertIn(CHAT, owner)
        self.assertNotIn(EMBEDDING, owner)
        self.assertNotIn("dall-e-2", owner)
        member = self.get(self.member).context["filters"].models
        self.assertIn(EMBEDDING, member)
        self.assertNotIn("dall-e-2", member)
        self.assertIn("dall-e-2", self.get(self.org_admin).context["filters"].models)

    def test_key_search_names_the_org_of_each_key_across_all_orgs(self):
        matches = self.matches(self.admin, "laptop", org="all")
        self.assertIn("solo-laptop · solo@example.com · lab-org", matches)
        self.assertIn("foreign-laptop · foreigner@example.com · other-org", matches)
        self.assertIn(
            "team-sa · Service account · NLP Lab · lab-org",
            self.matches(self.admin, "team", org="all"),
        )
        matches = self.matches(self.admin, "laptop", org=self.org.id)
        self.assertIn("solo-laptop · solo@example.com", matches)
        self.assertNotIn("lab-org", " ".join(matches))

    def test_all_orgs_finds_a_key_whose_owner_has_no_profile(self):
        root = User.objects.create_superuser("root", "root@example.com", "pw")
        self.log(make_token(root, "root-laptop"), "/profileless")
        self.assertIn("/profileless", self.get(self.admin, org="all").content.decode())
        self.assertEqual(
            self.matches(self.admin, "root", org="all"),
            ["root-laptop · root@example.com · No organization"],
        )

    def test_key_search_finds_only_keys_the_viewer_may_see(self):
        make_token(self.solo, "owner-spare")
        cases = (
            (self.solo, "laptop", ["solo-laptop · Personal"]),
            (self.solo, "other@example", []),
            (self.member, "sa", ["team-sa · Service account · NLP Lab"]),
            (self.sa_creator, "sa", []),
            (self.other_org_admin, "laptop", ["foreign-laptop · foreigner@example.com"]),
            (self.other_org_admin, "owner", []),
            (self.org_admin, "OTHER@", ["other-laptop · other@example.com"]),
            (
                self.org_admin,
                "sa-creator@",
                ["side-sa · Service account · Robotics", "team-sa · Service account · NLP Lab"],
            ),
            (self.member, "sa-creator@", []),
        )
        for user, query, expected in cases:
            with self.subTest(user=user.username, query=query):
                self.assertEqual(self.matches(user, query), expected)

    def test_key_search_opens_the_picker_and_keeps_the_other_filters(self):
        response = self.get(self.org_admin, key_q="laptop", model=CHAT, span="1d", before="1,1")
        self.assertContains(response, '<details class="dropdown" open>', count=1)
        self.assertContains(response, '<input type="hidden" name="model" value="gpt-4.1-nano">')
        self.assertContains(response, '<input type="hidden" name="span" value="1d">')
        self.assertNotContains(response, 'type="hidden" name="before"')
        self.assertContains(
            response, f'href="?model={CHAT}&amp;span=1d&amp;key={self.solo_key.id}"'
        )

    def test_picker_lists_the_scopes_and_the_viewers_own_keys(self):
        admin_key = make_token(self.org_admin, "admin-laptop")
        TeamMembership.objects.create(user_profile=self.org_admin.profile, team=self.side_team)
        filters = self.get(self.org_admin).context["filters"]
        self.assertEqual([key.name for key in filters.keys], ["admin-laptop", "side-sa"])
        response = self.get(self.org_admin, key=admin_key.id)
        self.assertContains(response, ">All keys</a>")
        self.assertContains(response, ">My keys</a>")
        self.assertNotContains(response, ">Everyone</a>")
        make_token(self.solo, "owner-spare")
        response = self.get(self.solo)
        self.assertContains(response, 'aria-current="true" class="menu-active">All keys</a>')
        self.assertNotContains(response, ">My keys</a>")

    def matches(self, user, query: str, **params) -> list[str]:
        filters = self.get(user, key_q=query, **params).context["filters"]
        return [key.name + " · " + key.note for key in filters.matches]

    def test_model_cell_filters_the_log_on_its_model(self):
        response = self.get(self.solo, status="2xx")
        self.assertContains(
            response, f'href="?status=2xx&amp;model={CHAT}" title="Show only {CHAT}"', count=1
        )

    def test_detail_opens_only_for_a_request_on_the_page(self):
        own = Request.objects.get(path="/own")
        own.user_agent = "openai-python/1.51"
        own.save()
        self.assertNotContains(self.get(self.solo), "openai-python/1.51")
        opened = self.get(self.solo, request=own.id)
        self.assertContains(opened, "openai-python/1.51")
        self.assertContains(opened, f'data-copy="{own.id}"')
        foreign = Request.objects.get(path="/foreign")
        foreign.user_agent = "foreign-agent/9"
        foreign.save()
        self.assertNotContains(self.get(self.solo, request=foreign.id), "foreign-agent/9")

    def test_detail_lists_only_fields_with_a_value(self):
        bare = self.log(self.solo_key, "/bare", user_agent="", output_tokens=0)
        content = self.get(self.solo, request=bare.id).content.decode()
        details = content[content.index("<dl") : content.index("</dl>")]
        self.assertIn("Request ID", details)
        for label in ("Error", "User agent", "User ID", "Gateway overhead", "Output speed"):
            self.assertNotIn(label, details)
        self.assertNotIn("n/a", details)

    def test_a_request_to_a_model_the_gateway_does_not_serve_says_so(self):
        typo = self.log(self.solo_key, "/typo", status_code=400, model="gpt-5")
        response = self.get(self.solo)
        row = next(r for r in response.context["log"].rows if r.request == typo)
        self.assertTrue(row.unknown_model)
        self.assertContains(response, "Not a model on this gateway.", count=2)
        self.assertContains(
            response,
            f'<a class="link-plain whitespace-nowrap" href="{reverse("models")}">Models</a>',
        )
        found = self.get(self.solo, model="gpt-5", request=typo.id)
        self.assertEqual([row.request for row in found.context["log"].rows], [typo])
        self.assertContains(found, "This gateway does not serve a model with that name.")

    def test_a_served_model_gets_no_such_explanation(self):
        self.assertNotContains(self.get(self.solo), "Not a model on this gateway.")

    def test_the_log_says_which_refusals_are_not_logged(self):
        self.assertContains(
            self.get(self.solo), "Requests the gateway stops at a limit are not logged."
        )

    def test_an_email_in_the_detail_wraps_before_its_at_sign(self):
        sent = self.log(self.solo_key, "/mail", user_id="someone.long@example.com")
        self.assertContains(self.get(self.solo, request=sent.id), "someone.long<wbr>@example.com")

    def test_detail_explains_a_failed_status_and_shows_the_rest(self):
        failed = self.log(
            self.solo_key,
            "/failed",
            status_code=429,
            user_id="alice",
            processing_time_ms=12,
            output_tokens=300,
            response_time_ms=2000,
        )
        content = self.get(self.solo, request=failed.id).content.decode()
        self.assertIn(
            '>Error</dt><dd class="text-sm">The model provider was over a limit '
            "and refused the request.</dd>",
            content,
        )
        self.assertIn(">User ID</dt>", content)
        self.assertIn(">alice</dd>", content)
        self.assertIn("150 tok/s", content)

    def test_gateway_overhead_shows_to_org_admins_and_admins_only(self):
        timed = self.log(self.team_key, "/timed", processing_time_ms=12)
        team_admin = self.team_admin
        for user, shown in (
            (self.member, False),
            (team_admin, False),
            (self.org_admin, True),
            (self.admin, True),
        ):
            with self.subTest(user=user.username):
                content = self.get(user, request=timed.id).content.decode()
                self.assertIn(">/timed</div>", content)
                self.assertEqual(">Gateway overhead</dt>" in content, shown)

    def test_key_column_shows_only_when_rows_may_differ_in_key(self):
        make_token(self.solo, "owner-desktop")
        side_member = self.user("side-member")
        join(side_member, self.side_team)
        for user, params, shown in (
            (side_member, {}, False),
            (self.member, {}, True),
            (self.other, {}, False),
            (self.solo, {}, True),
            (self.solo, {"key": self.solo_key.id}, False),
            (self.org_admin, {}, True),
            (self.org_admin, {"key": self.solo_key.id}, False),
        ):
            with self.subTest(user=user.username, **params):
                content = self.get(user, **params).content.decode()
                self.assertEqual('<th rowspan="2">Key</th>' in content, shown)


class RequestsSearchTest(RequestsPageTestBase):
    def test_search_matches_the_id_path_user_id_and_user_agent(self):
        own = Request.objects.get(path="/own")
        self.log(self.solo_key, "/agent", user_agent="Curl/8.5")
        self.log(self.solo_key, "/sent", user_id="Alice@Example.com")
        for q, paths in (
            (str(own.id), ["/own"]),
            ("OWN", ["/own"]),
            ("curl", ["/agent"]),
            (" alice@ ", ["/sent"]),
            ("nothing-like-it", []),
            ("o\x00wn", ["/own"]),
        ):
            with self.subTest(q=q):
                response = self.get(self.solo, q=q)
                self.assertEqual(listed(response), paths)
                self.assertEqual(response.context["log"].tabs[0].count, len(paths))

    def test_search_never_reaches_beyond_what_the_viewer_sees(self):
        foreign = Request.objects.get(path="/foreign")
        for user, params, paths in (
            (self.solo, {"q": str(foreign.id)}, []),
            (self.solo, {"q": "others"}, []),
            (self.solo, {"q": "solo@example.com"}, []),
            (self.member, {"q": "sa"}, ["/team-sa"]),
            (self.org_admin, {"q": "foreign"}, []),
            (self.other_org_admin, {"q": str(foreign.id)}, ["/foreign"]),
            (self.admin, {"q": "foreign"}, []),
        ):
            with self.subTest(user=user.username, **params):
                self.assertEqual(listed(self.get(user, **params)), paths)

    def test_search_stays_within_the_filters_and_the_time_frame(self):
        self.log(self.solo_key, "/find-old", ago=timedelta(days=3))
        self.log(self.solo_key, "/find-failed", status_code=500)
        self.log(self.solo_key, "/find-other-model", model=EMBEDDING)
        self.assertEqual(len(listed(self.get(self.solo, q="find", span="1w"))), 3)
        for params, paths in (
            ({}, ["/find-failed", "/find-other-model"]),
            ({"status": "failed"}, ["/find-failed"]),
            ({"model": EMBEDDING}, ["/find-other-model"]),
        ):
            with self.subTest(**params):
                self.assertEqual(sorted(listed(self.get(self.solo, q="find", **params))), paths)

    def test_search_form_keeps_the_filters_and_starts_on_the_first_page(self):
        content = self.get(
            self.org_admin, model=CHAT, status="failed", sort="time", before="1,1", q="x"
        ).content.decode()
        forms = [part.split("</form>", 1)[0] for part in content.split('role="search"')[1:]]
        form = next(form for form in forms if 'name="q"' in form)
        for name, value in (("model", CHAT), ("status", "failed"), ("sort", "time")):
            self.assertIn(f'name="{name}" value="{value}"', form)
        self.assertNotIn('name="before"', form)
        self.assertIn('name="q" value="x"', form)

    def test_text_search_needs_a_bounded_scan_and_an_id_search_does_not(self):
        own = Request.objects.get(path="/own")
        team_admin = self.team_admin
        for user, params, searches_text in (
            (self.org_admin, {"span": "1m"}, False),
            (self.org_admin, {"span": "1w"}, True),
            (self.org_admin, {"span": "1m", "key": self.solo_key.id}, True),
            (self.org_admin, {"span": "1m", "model": CHAT}, True),
            (self.org_admin, {"span": "1m", "scope": "mine"}, True),
            (self.solo, {"span": "1m"}, True),
            (team_admin, {"span": "1m"}, True),
        ):
            with self.subTest(user=user.username, **params):
                response = self.get(user, **params, q="/own")
                content = response.content.decode()
                self.assertEqual(response.context["log"].searches_text, searches_text)
                if searches_text:
                    self.assertIn('aria-label="Search requests"', content)
                    self.assertNotIn("to search text", content)
                    continue
                self.assertEqual(listed(response), [])
                self.assertIn('placeholder="Request ID" aria-label="Request ID"', content)
                self.assertIn("Narrow to 7 days or pick a key or model to search text.", content)
                self.assertNotIn("to search text", self.get(user, **params).content.decode())
                by_id = self.get(user, **params, q=str(own.id))
                self.assertEqual(listed(by_id), ["/own"])
                self.assertEqual(by_id.context["log"].tabs[0].count, 1)
                self.assertNotIn("to search text", by_id.content.decode())

    def test_a_blocked_text_search_leaves_the_org_wide_counts_alone(self):
        wide = {"span": "1m"}
        total = self.get(self.org_admin, **wide).context["log"].tabs[0].count
        self.assertEqual(self.get(self.org_admin, **wide, q="/own").context["log"].tabs[0].count, 0)
        self.assertEqual(self.get(self.org_admin, **wide).context["log"].tabs[0].count, total)
        self.assertEqual(
            self.get(self.org_admin, span="1w", q="/own").context["log"].tabs[0].count, 1
        )


class RequestsSortTest(RequestsPageTestBase):
    def setUp(self):
        super().setUp()
        Request.objects.filter(path="/own").delete()
        fields = (
            (300, 10, 5, 200, CHAT),
            (None, 1, 1, 500, EMBEDDING),
            (300, 2, 2, 429, CHAT),
            (100, 40, 0, None, EMBEDDING),
            (200, 3, 1, 404, CHAT),
        )
        for n, (ms, input_tokens, output_tokens, status, model) in enumerate(fields):
            self.log(
                self.solo_key,
                f"/sorted-{n}",
                ago=timedelta(minutes=n),
                response_time_ms=ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status_code=status,
                model=model,
            )

    def test_each_column_sorts_both_ways_with_missing_values_last(self):
        for sort, order in (
            ("-time", [0, 1, 2, 3, 4]),
            ("time", [4, 3, 2, 1, 0]),
            ("-response", [2, 0, 4, 3, 1]),
            ("response", [3, 4, 0, 2, 1]),
            ("-tokens", [3, 0, 4, 2, 1]),
            ("tokens", [1, 2, 4, 0, 3]),
            ("model", [0, 2, 4, 1, 3]),
            ("-status", [1, 2, 4, 0, 3]),
        ):
            with self.subTest(sort=sort):
                expected = [f"/sorted-{n}" for n in order]
                self.assertEqual(listed(self.get(self.solo, sort=sort)), expected)

    def test_org_wide_30_days_sorts_by_time_only(self):
        for user, params in ((self.org_admin, {"span": "1m"}), (self.admin, {"span": "1m"})):
            with self.subTest(user=user.username, **params):
                response = self.get(user, sort="-response", **params)
                self.assertEqual(str(response.context["log"].sort), "-time")
                content = response.content.decode()
                self.assertRegex(content, r'class="text-right">Response\s*</th>')
                self.assertNotIn("To sort by", content)
                self.assertIn("sort=time", content)
                oldest = self.get(user, sort="time", **params).context["log"].rows[0]
                self.assertEqual(oldest.request.path, "/others")

    def test_a_short_time_frame_a_key_a_model_or_own_keys_allow_every_sort(self):
        for params in (
            {},
            {"span": "1w"},
            *(
                {"span": "1m", narrower: value}
                for narrower, value in (
                    ("model", CHAT),
                    ("key", self.solo_key.id),
                    ("scope", "mine"),
                )
            ),
        ):
            with self.subTest(**params):
                response = self.get(self.org_admin, sort="-response", **params)
                self.assertEqual(str(response.context["log"].sort), "-response")
                self.assertIn('">Response<svg', response.content.decode())

    def test_headings_mark_the_sort_and_toggle_it(self):
        content = self.get(self.solo).content.decode()
        self.assertEqual(content.count("aria-sort="), 1)
        self.assertIn('aria-sort="descending"><a', content)
        self.assertIn("?sort=time", content)
        content = self.get(self.solo, sort="-response").content.decode()
        self.assertEqual(content.count("aria-sort="), 1)
        self.assertIn('text-right" aria-sort="descending"', content)
        self.assertIn("?sort=response", content)
        content = self.get(self.solo, sort="model").content.decode()
        self.assertIn('aria-sort="ascending"', content)
        self.assertIn("?sort=-model", content)
