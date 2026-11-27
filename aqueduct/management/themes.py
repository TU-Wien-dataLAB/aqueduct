"""Themes beside light and dark: the built-in ones in builtin_themes/ that BUILTIN_LIGHT_THEMES and
BUILTIN_DARK_THEMES name, in their order, then an operator's in THEMES_DIR, which replace a
built-in theme of the same name. DEFAULT_LIGHT_THEME and DEFAULT_DARK_THEME name the theme of
each scheme for people who have not picked one.

A file `<id>.css` holds one `[data-theme="<id>"]` rule that declares `color-scheme: light` or
`dark`. The page starts from that scheme's variables, so the rule sets only the colors, radii and
effects it changes. The page inlines each file, so a theme needs no collectstatic. A file that
breaks a rule is logged and left out.
"""

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

log = logging.getLogger("aqueduct")

BUILT_IN = ("system", "light", "dark")
BUILTIN_THEMES_DIR = Path(__file__).resolve().parent / "builtin_themes"
_ID = re.compile(r"[A-Za-z0-9_-]+")
_SCHEME = re.compile(r"color-scheme\s*:\s*(light|dark)\b")


@dataclass(frozen=True)
class Theme:
    id: str
    label: str
    scheme: str
    css: str


def custom_themes() -> tuple[Theme, ...]:
    return _themes(
        tuple(settings.BUILTIN_LIGHT_THEMES),
        tuple(settings.BUILTIN_DARK_THEMES),
        getattr(settings, "THEMES_DIR", None),
    )


def default_themes() -> dict[str, str]:
    """The theme of each scheme for people who have not picked one: light or dark, or the
    offered theme of that scheme that DEFAULT_LIGHT_THEME or DEFAULT_DARK_THEME names."""
    return _defaults(settings.DEFAULT_LIGHT_THEME, settings.DEFAULT_DARK_THEME, custom_themes())


@functools.cache
def _themes(
    light: tuple[str, ...], dark: tuple[str, ...], directory: str | None
) -> tuple[Theme, ...]:
    built_in = {theme.id: theme for theme in _load(str(BUILTIN_THEMES_DIR))}
    offered = []
    for scheme, theme_ids in (("light", light), ("dark", dark)):
        for theme_id in theme_ids:
            theme = built_in.get(theme_id)
            if theme and theme.scheme == scheme:
                offered.append(theme)
            else:
                log.error("Theme %s left out: it is not a built-in %s theme.", theme_id, scheme)
    themes = (*offered, *(_load(str(directory)) if directory else ()))
    return tuple({theme.id: theme for theme in themes}.values())


@functools.cache
def _defaults(light: str, dark: str, themes: tuple[Theme, ...]) -> dict[str, str]:
    defaults = {}
    for scheme, theme_id in (("light", light), ("dark", dark)):
        offered = {scheme} | {theme.id for theme in themes if theme.scheme == scheme}
        if theme_id in offered:
            defaults[scheme] = theme_id
        else:
            log.error("Default %s theme %s ignored: it is not offered.", scheme, theme_id)
            defaults[scheme] = scheme
    return defaults


@functools.cache
def _load(directory: str) -> tuple[Theme, ...]:
    path = Path(directory)
    if not path.is_dir():
        log.error("Theme directory %s is not a directory; no themes loaded from it.", directory)
        return ()
    themes = []
    for file in sorted(path.glob("*.css")):
        theme_id = file.stem
        css = file.read_text(encoding="utf-8")
        scheme = _SCHEME.search(css)
        if not _ID.fullmatch(theme_id) or theme_id in BUILT_IN:
            problem = "its name must be letters, digits, - or _, and not system, light or dark"
        elif not re.search(rf"\[data-theme=[\"']?{theme_id}[\"']?\]", css):
            problem = f'it has no [data-theme="{theme_id}"] rule'
        elif not scheme:
            problem = "it declares no color-scheme: light or dark"
        elif "</" in css:
            problem = "it contains </, which would end the page's <style> element"
        else:
            label = re.sub(r"[-_]+", " ", theme_id)
            label = label[:1].upper() + label[1:]
            themes.append(Theme(id=theme_id, label=label, scheme=scheme[1], css=css))
            continue
        log.error("Theme %s left out: %s.", file, problem)
    return tuple(themes)
