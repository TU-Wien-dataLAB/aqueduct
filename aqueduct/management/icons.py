from enum import Enum, unique


def _el(tag: str, **attrs: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (tag, tuple(attrs.items()))


@unique
class Icon(Enum):
    OVERVIEW = (_el("path", d="M5 20V11M12 20V4M19 20v-6M3 20h18"),)
    MODELS = (
        _el("path", d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"),
        _el("path", d="M12 12l8-4.5M12 12v9M12 12L4 7.5"),
    )
    MCP = (_el("path", d="M9 3v5M15 3v5M6 8h12v3a6 6 0 01-12 0V8zM12 17v4"),)
    KEY = (_el("circle", cx="8", cy="15", r="4"), _el("path", d="M11 12l9-9M17 6l3 3M14 9l2 2"))
    REQUESTS = (_el("path", d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"),)
    TEAMS = (
        _el("circle", cx="9", cy="8", r="3.5"),
        _el(
            "path",
            d=(
                "M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6M16 4.6a3.5 3.5 0 010 6.8"
                "M18 14.3c2.1.7 3.5 2.6 3.5 5.7"
            ),
        ),
    )
    FILE = (
        _el("path", d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5z"),
        _el("path", d="M14 3v5h5"),
    )
    BATCH = (_el("path", d="M12 3l9 5-9 5-9-5 9-5z"), _el("path", d="M3 13l9 5 9-5"))
    VECTOR_STORE = (
        _el("ellipse", cx="12", cy="6", rx="8", ry="3"),
        _el("path", d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"),
    )
    ORGANIZATION = (
        _el(
            "path",
            d=(
                "M4 21V5a1 1 0 011-1h9a1 1 0 011 1v16M15 10h4a1 1 0 011 1v10M3 21h18"
                "M8 8h3M8 12h3M8 16h3"
            ),
        ),
    )
    SETTINGS = (
        _el("path", d="M4 7h9M17 7h3M4 17h3M11 17h9"),
        _el("circle", cx="15", cy="7", r="2"),
        _el("circle", cx="9", cy="17", r="2"),
    )
    ADMIN = (_el("path", d="M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4"),)
    PROFILING = (_el("path", d="M3 12h4l3-7 4 14 3-7h4"),)
    LINK = (
        _el(
            "path",
            d=(
                "M10 14a4 4 0 005.66 0l3-3a4 4 0 00-5.66-5.66l-1 1"
                "M14 10a4 4 0 00-5.66 0l-3 3a4 4 0 005.66 5.66l1-1"
            ),
        ),
    )
    EXTERNAL = (
        _el("path", d="M14 5h5v5M19 5l-8 8M11 7H6a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1v-5"),
    )
    SEARCH = (_el("circle", cx="11", cy="11", r="6.5"), _el("path", d="M16 16l4.5 4.5"))
    WARNING = (_el("path", d="M12 4l9 16H3l9-16zM12 10v5M12 17.5v.5"),)
    OFF = (_el("circle", cx="12", cy="12", r="9"), _el("path", d="M8 12h8"))
    PLUS = (_el("path", d="M12 5v14M5 12h14"),)
    CHEVRON = (_el("path", d="M9 6l6 6-6 6"),)
    SORT = (_el("path", d="M8 9l4-4 4 4M8 15l4 4 4-4"),)
    TRANSFER = (_el("path", d="M4 8h14M14 4l4 4-4 4M20 16H6M10 12l-4 4 4 4"),)
    ROTATE = (_el("path", d="M20 12a8 8 0 11-2.6-5.9M20 4v5h-5"),)
    DELETE = (_el("path", d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"),)
    OK = (_el("circle", cx="12", cy="12", r="8.5"), _el("path", d="M8.5 12.5l2.5 2.5 4.5-5"))
    PENDING = (_el("circle", cx="12", cy="12", r="8.5"), _el("path", d="M12 7.5V12l3 2"))
    FAILED = (_el("circle", cx="12", cy="12", r="8.5"), _el("path", d="M9 9l6 6M15 9l-6 6"))
    INFO = (_el("circle", cx="12", cy="12", r="8.5"), _el("path", d="M12 11v5M12 8v.01"))
    CLOSE = (_el("path", d="M6 6l12 12M18 6L6 18"),)
    MENU = (_el("path", d="M4 7h16M4 12h16M4 17h16"),)
    COPY = (
        _el("rect", x="9", y="9", width="11", height="11", rx="2"),
        _el("path", d="M5 15V6a2 2 0 012-2h9"),
    )
    SUN = (
        _el("circle", cx="12", cy="12", r="4"),
        _el(
            "path",
            d=(
                "M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4"
                "M5.6 18.4l1.4-1.4M17 7l1.4-1.4"
            ),
        ),
    )
    MOON = (_el("path", d="M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z"),)
    USER = (
        _el("circle", cx="12", cy="9", r="4"),
        _el("path", d="M4.5 20.5c1-3.4 4-5.5 7.5-5.5s6.5 2.1 7.5 5.5"),
    )
    MATRIX = (
        _el("path", d="M7 4H4v16h3M17 4h3v16h-3M8 15v-5M8 12a2 2 0 014 0v3M12 12a2 2 0 014 0v3"),
    )
    GITHUB = (
        _el(
            "path",
            d=(
                "M15 22v-4a4.8 4.8 0 00-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5"
                ".28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2"
                "c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 004 9c0 3.5 3 5.5 6 5.5"
                "-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"
            ),
        ),
        _el("path", d="M9 18c-4.51 2-5-2-7-2"),
    )
    SIGN_OUT = (_el("path", d="M10 4H6a2 2 0 00-2 2v12a2 2 0 002 2h4M15 8l4 4-4 4M19 12H9"),)
