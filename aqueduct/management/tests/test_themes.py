import tempfile
from pathlib import Path

from django.test import override_settings
from django.urls import reverse

from management.tests.helpers import RoleWorld
from management.themes import custom_themes

TU_WIEN = """[data-theme="TU-Wien"] {
  color-scheme: light;
  --color-primary: #006699;
  --radius-box: .25rem;
}
"""
BUILT_IN = [
    ("black", "dark"),
    ("cupcake", "light"),
    ("night", "dark"),
    ("nord", "light"),
    ("phosphor", "dark"),
    ("riso", "light"),
    ("synthwave", "dark"),
    ("wireframe", "light"),
]


def ids(themes) -> list[str]:
    return [theme.id for theme in themes]


class ThemesDirTest(RoleWorld):
    def themes_dir(self, files: dict[str, str]) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name, css in files.items():
            (Path(directory.name) / name).write_text(css)
        return directory.name

    def test_a_theme_file_becomes_a_theme_named_after_it(self):
        with override_settings(
            THEMES_DIR=self.themes_dir({"TU-Wien.css": TU_WIEN, "notes.txt": "not a theme"})
        ):
            *built_in, theme = custom_themes()
        self.assertEqual(ids(built_in), [theme_id for theme_id, _ in BUILT_IN])
        self.assertEqual(
            (theme.id, theme.label, theme.scheme, theme.css),
            ("TU-Wien", "TU Wien", "light", TU_WIEN),
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

    def test_daisyui_themes_are_built_in_with_their_scheme(self):
        with self.assertNoLogs("aqueduct", "ERROR"):
            themes = custom_themes()
        self.assertEqual([(theme.id, theme.scheme) for theme in themes], BUILT_IN)
        self.assertEqual(themes[0].label, "Black")

    def test_an_operator_theme_replaces_the_built_in_one_of_its_name(self):
        nord = '[data-theme="nord"] { color-scheme: dark; }'
        with override_settings(THEMES_DIR=self.themes_dir({"nord.css": nord})):
            themes = {theme.id: theme for theme in custom_themes()}
        self.assertEqual(len(themes), len(BUILT_IN))
        self.assertEqual((themes["nord"].css, themes["nord"].scheme), (nord, "dark"))

    def test_every_page_inlines_the_themes_and_settings_offers_them(self):
        with override_settings(THEMES_DIR=self.themes_dir({"TU-Wien.css": TU_WIEN})):
            page = self.as_user(self.solo).get(reverse("tokens"))
            self.assertContains(
                page,
                f'<style data-theme-id="TU-Wien" data-scheme="light">{TU_WIEN}</style>',
                html=False,
            )
            settings_page = self.as_user(self.solo).get(reverse("settings"))
        self.assertContains(settings_page, 'name="light-theme" value="TU-Wien"')
        self.assertContains(settings_page, "TU Wien")
        self.assertContains(settings_page, 'data-theme="TU-Wien" data-scheme="light"')

    def test_without_a_themes_dir_settings_offers_a_mode_and_the_built_in_theme_of_each_scheme(
        self,
    ):
        page = self.as_user(self.solo).get(reverse("settings"))
        for mode in ("system", "light", "dark"):
            self.assertContains(page, f'name="mode" value="{mode}"')
        for theme_id, scheme in [("light", "light"), ("dark", "dark"), *BUILT_IN]:
            self.assertContains(page, f'name="{scheme}-theme" value="{theme_id}"')
        self.assertContains(page, '<style data-theme-id="night" data-scheme="dark">')
