import tempfile
from pathlib import Path

from django.test import override_settings
from django.urls import reverse

from management import themes as themes_module
from management.tests.helpers import RoleWorld
from management.themes import custom_themes, default_themes

AQUEDUCT_BLUE = """[data-theme="Aqueduct-Blue"] {
  color-scheme: light;
  --color-primary: #006699;
  --radius-box: .25rem;
}
"""
BUILT_IN = [
    ("wireframe", "light"),
    ("nord", "light"),
    ("cupcake", "light"),
    ("riso", "light"),
    ("black", "dark"),
    ("night", "dark"),
    ("synthwave", "dark"),
    ("phosphor", "dark"),
]


def ids(themes) -> list[str]:
    return [theme.id for theme in themes]


class ThemesDirTest(RoleWorld):
    def setUp(self):
        super().setUp()
        # Each cache logs a problem only once, so every test starts without them.
        for cached in (themes_module._load, themes_module._themes, themes_module._defaults):
            cached.cache_clear()

    def themes_dir(self, files: dict[str, str]) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name, css in files.items():
            (Path(directory.name) / name).write_text(css)
        return directory.name

    def test_a_theme_file_becomes_a_theme_named_after_it(self):
        with override_settings(
            THEMES_DIR=self.themes_dir(
                {"Aqueduct-Blue.css": AQUEDUCT_BLUE, "notes.txt": "not a theme"}
            )
        ):
            *built_in, theme = custom_themes()
        self.assertEqual(ids(built_in), [theme_id for theme_id, _ in BUILT_IN])
        self.assertEqual(
            (theme.id, theme.label, theme.scheme, theme.css),
            ("Aqueduct-Blue", "Aqueduct Blue", "light", AQUEDUCT_BLUE),
        )

    def test_files_that_break_a_rule_are_logged_and_left_out(self):
        files = {
            "dark.css": '[data-theme="dark"] { color-scheme: dark; }',
            "no-rule.css": "html { color-scheme: dark; }",
            "no-scheme.css": '[data-theme="no-scheme"] { --color-primary: red; }',
            "closes-style.css": '[data-theme="closes-style"] { color-scheme: dark; } </style>',
            "dots.in.name.css": '[data-theme="dots.in.name"] { color-scheme: dark; }',
        }
        with (
            override_settings(THEMES_DIR=self.themes_dir(files)),
            self.assertLogs("aqueduct", "ERROR") as logs,
        ):
            self.assertEqual(ids(custom_themes()), [theme_id for theme_id, _ in BUILT_IN])
        self.assertEqual(len(logs.output), len(files))

    def test_a_missing_directory_is_logged_and_gives_no_themes(self):
        with (
            override_settings(THEMES_DIR="/nonexistent/themes"),
            self.assertLogs("aqueduct", "ERROR"),
        ):
            self.assertEqual(ids(custom_themes()), [theme_id for theme_id, _ in BUILT_IN])

    def test_built_in_themes_pair_each_light_theme_with_a_dark_one(self):
        with self.assertNoLogs("aqueduct", "ERROR"):
            themes = custom_themes()
        self.assertEqual([(theme.id, theme.scheme) for theme in themes], BUILT_IN)
        self.assertEqual(themes[4].label, "Black")

    def test_the_built_in_lists_pick_and_order_the_built_in_themes(self):
        with override_settings(BUILTIN_LIGHT_THEMES=["riso", "nord"], BUILTIN_DARK_THEMES=[]):
            self.assertEqual(ids(custom_themes()), ["riso", "nord"])
            page = self.as_user(self.solo).get(reverse("settings"))
        self.assertContains(page, 'name="light-theme" value="light"')
        self.assertContains(page, 'name="dark-theme" value="dark"')
        self.assertNotContains(page, 'value="wireframe"')

    def test_a_listed_name_that_is_no_built_in_theme_of_its_scheme_is_logged_and_left_out(self):
        with (
            override_settings(
                BUILTIN_LIGHT_THEMES=["nrod", "night", "nord"], BUILTIN_DARK_THEMES=[]
            ),
            self.assertLogs("aqueduct", "ERROR") as logs,
        ):
            self.assertEqual(ids(custom_themes()), ["nord"])
        self.assertEqual(len(logs.output), 2)

    def test_without_defaults_each_scheme_falls_back_to_light_or_dark(self):
        self.assertEqual(default_themes(), {"light": "light", "dark": "dark"})
        page = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(page, 'data-light-theme="light" data-dark-theme="dark"')

    def test_a_default_may_name_a_built_in_or_an_operator_theme(self):
        with override_settings(
            THEMES_DIR=self.themes_dir({"Aqueduct-Blue.css": AQUEDUCT_BLUE}),
            DEFAULT_LIGHT_THEME="Aqueduct-Blue",
            DEFAULT_DARK_THEME="night",
        ):
            self.assertEqual(default_themes(), {"light": "Aqueduct-Blue", "dark": "night"})
            page = self.as_user(self.solo).get(reverse("tokens"))
        self.assertContains(page, 'data-light-theme="Aqueduct-Blue" data-dark-theme="night"')

    def test_a_default_that_is_not_offered_in_its_scheme_is_logged_and_ignored(self):
        with (
            override_settings(
                BUILTIN_DARK_THEMES=["night"],
                DEFAULT_LIGHT_THEME="night",
                DEFAULT_DARK_THEME="black",
            ),
            self.assertLogs("aqueduct", "ERROR") as logs,
        ):
            self.assertEqual(default_themes(), {"light": "light", "dark": "dark"})
        self.assertEqual(len(logs.output), 2)

    def test_an_operator_theme_replaces_the_built_in_one_of_its_name(self):
        nord = '[data-theme="nord"] { color-scheme: dark; }'
        with override_settings(THEMES_DIR=self.themes_dir({"nord.css": nord})):
            themes = {theme.id: theme for theme in custom_themes()}
        self.assertEqual(len(themes), len(BUILT_IN))
        self.assertEqual((themes["nord"].css, themes["nord"].scheme), (nord, "dark"))

    def test_every_page_inlines_the_themes_and_settings_offers_them(self):
        with override_settings(THEMES_DIR=self.themes_dir({"Aqueduct-Blue.css": AQUEDUCT_BLUE})):
            page = self.as_user(self.solo).get(reverse("tokens"))
            self.assertContains(
                page,
                f'<style data-theme-id="Aqueduct-Blue" data-scheme="light">{AQUEDUCT_BLUE}</style>',
                html=False,
            )
            settings_page = self.as_user(self.solo).get(reverse("settings"))
        self.assertContains(settings_page, 'name="light-theme" value="Aqueduct-Blue"')
        self.assertContains(settings_page, "Aqueduct Blue")
        self.assertContains(settings_page, 'data-theme="Aqueduct-Blue" data-scheme="light"')

    def test_without_a_themes_dir_settings_offers_a_mode_and_the_built_in_theme_of_each_scheme(
        self,
    ):
        page = self.as_user(self.solo).get(reverse("settings"))
        for mode in ("system", "light", "dark"):
            self.assertContains(page, f'name="mode" value="{mode}"')
        for theme_id, scheme in [("light", "light"), ("dark", "dark"), *BUILT_IN]:
            self.assertContains(page, f'name="{scheme}-theme" value="{theme_id}"')
        self.assertContains(page, '<style data-theme-id="night" data-scheme="dark">')
