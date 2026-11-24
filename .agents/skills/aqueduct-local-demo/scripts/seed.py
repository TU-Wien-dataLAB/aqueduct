"""Define the local demo's users: one per role, with teams, keys, and ten days of requests.

start.sh pipes this into `manage.py shell` on every start, before Dex starts. It rewrites the
login snippet, the Dex logins (appended to `DEX_CONFIG`), and the Dex login page (`DEX_PAGE`),
which shows one button per user. A person's teams, keys, and requests are added only when this
script creates the person, so restarts keep the data. Prints the users for the hand-off.
"""

# Standalone script piped into `manage.py shell`, not a package; its numbers are made up.
# ruff: noqa: INP001, T201, S311, PLR2004

import os
import random
from datetime import timedelta
from html import escape
from pathlib import Path

from django.contrib.auth import get_user_model
from django.utils import timezone

from gateway.config import get_router_config
from management.models import (
    Org,
    Request,
    ServiceAccount,
    Snippet,
    SnippetType,
    Team,
    TeamMembership,
    Token,
    UserProfile,
)

PEOPLE = [
    {"email": "admin@example.com", "role": "admin", "note": "platform admin, switches orgs"},
    {
        "email": "you@example.com",
        "role": "org-admin",
        "note": "org admin of default, admin of Demo Team",
        "teams": {"Demo Team": True},
        "keys": ["you-laptop"],
        "service_keys": {"eval-pipeline": "Demo Team"},
    },
    {
        "email": "ada@example.com",
        "role": "user",
        "note": "member of Demo Team",
        "teams": {"Demo Team": False},
        "keys": ["ada-laptop", "ada-notebook"],
    },
    {
        "email": "grace@example.com",
        "role": "user",
        "note": "admin of Vision Lab",
        "teams": {"Vision Lab": True},
        "keys": ["grace-cluster"],
        "service_keys": {"captioner": "Vision Lab"},
    },
    {"email": "new@example.com", "role": "user", "note": "no keys yet: first-key card"},
    {
        "email": "marie@example.com",
        "role": "user",
        "org": "physics",
        "note": "only user of the second org",
        "keys": ["marie-workstation"],
    },
]
PATHS = {
    "completion": "/v1/completions",
    "response": "/v1/responses",
    "embedding": "/v1/embeddings",
    "rerank": "/v1/rerank",
    "audio_speech": "/v1/audio/speech",
    "audio_transcription": "/v1/audio/transcriptions",
    "image_generation": "/v1/images/generations",
}
"""The path of each router model mode, without its plural `s`; chat is the default."""
AGENTS = ["OpenAI/Python 1.82.0", "curl/8.7.1", "python-httpx/0.28.1", "Open WebUI"]
PASSWORD, PASSWORD_HASH = "1234", "$2a$12$dvOAiVxM0UtJeMD9oId1bO8fYsymfMc/xZmFzbYdnpbmdc1T5LK9q"
DEX_PAGE = """{{ template "header.html" . }}
<div class="theme-panel">
  <h2 class="theme-heading">Sign in as</h2>
  {{ if .Invalid }}
    <div id="login-error" class="dex-error-box">Invalid {{ .UsernamePrompt }} and password.</div>
  {{ end }}
  {buttons}
</div>
{{ template "footer.html" . }}
"""
"""Replaces Dex's password page; each button posts one user's login to Dex's own form URL."""
DEX_BUTTON = """
  <form class="theme-form-row" method="post" action="{{ .PostURL }}">
    <input type="hidden" name="login" value="{email}">
    <input type="hidden" name="password" value="{password}">
    <button class="dex-btn theme-btn-provider"><span class="dex-btn-text">{email}</span></button>
    <div class="dex-subtle-text">{role} · {note}</div>
  </form>"""

User = get_user_model()
rng = random.Random(7)
now = timezone.now()

roles = {person["email"]: person["role"] for person in PEOPLE}
orgs = {person["email"]: person.get("org", "default") for person in PEOPLE}
Snippet.objects.filter(type=SnippetType.CONFIG, active=True).update(active=False)
Snippet.objects.create(
    name="Local demo roles",
    type=SnippetType.CONFIG,
    active=True,
    code=(
        "class LocalDemoConfig(ConfigSnippet):\n"
        "    def org_name(self, claims):\n"
        f"        return {orgs!r}.get(claims.get('email'), 'default')\n\n"
        "    def user_group(self, claims):\n"
        f"        return {roles!r}.get(claims.get('email'), 'user')\n"
    ),
)

buttons = "".join(
    DEX_BUTTON.replace("{password}", PASSWORD)
    .replace("{email}", escape(person["email"]))
    .replace("{role}", escape(person["role"]))
    .replace("{note}", escape(person["note"]))
    for person in PEOPLE
)
Path(os.environ["DEX_PAGE"]).write_text(DEX_PAGE.replace("{buttons}", buttons))
with Path(os.environ["DEX_CONFIG"]).open("a") as config:
    for number, person in enumerate(PEOPLE, 1):
        email = person["email"]
        config.write(
            f'  - {{email: "{email}", username: "{email.split("@")[0]}", userID: "{number}", '
            f'hash: "{PASSWORD_HASH}"}}\n'
        )

models = {}
for model in get_router_config().get("model_list", []):
    mode = (model.get("model_info") or {}).get("mode") or "chat"
    models[model["model_name"]] = PATHS.get(mode.removesuffix("s"), "/v1/chat/completions")
weights = [4 if path.startswith("/v1/chat") else 1 for path in models.values()]


def add_requests(token: Token) -> None:
    """Give the key a few to a few dozen requests on each of the last ten days."""
    agent, rate = rng.choice(AGENTS), rng.uniform(0.4, 1.5)
    requests = []
    for day in range(10):
        for _ in range(int(rng.randint(2, 30) * rate * (1.6 if day < 3 else 1))):
            model, path = rng.choices(list(models.items()), weights=weights)[0]
            status = rng.choices([200, 429, 400, 500, 503], weights=[88, 5, 3, 2, 2])[0]
            ok, text = status < 400, path.endswith(("completions", "responses"))
            input_tokens = rng.randint(20, 4000) if ok and "images" not in path else 0
            output_tokens = rng.randint(10, 1500) if ok and text else 0
            response_ms = rng.randint(80, 9000) if ok else rng.randint(5, 300)
            requests.append(
                Request(
                    token=token,
                    model=model,
                    path=path,
                    method="POST",
                    status_code=status,
                    timestamp=now - timedelta(days=day, minutes=rng.randint(0, 1439)),
                    input_tokens=input_tokens,
                    cached_input_tokens=input_tokens // 3 if text and rng.random() < 0.3 else 0,
                    output_tokens=output_tokens,
                    reasoning_tokens=output_tokens // 2 if text and rng.random() < 0.3 else 0,
                    response_time_ms=response_ms,
                    processing_time_ms=response_ms // 20 if ok else None,
                    user_agent=agent,
                    ip_address="127.0.0.1",
                    user_id=f"user-{rng.randint(1, 5)}" if token.service_account_id else "",
                )
            )
    Request.objects.bulk_create(requests)


def add_key(user: User, name: str, team: Team | None = None) -> None:
    service_account = ServiceAccount.objects.create(name=name, team=team) if team else None
    token = Token(name=name, user=user, service_account=service_account)
    token._set_new_key()
    token.save()
    if models:
        add_requests(token)


for person in PEOPLE:
    email = person["email"]
    user, created = User.objects.get_or_create(email=email, defaults={"username": email})
    if created:
        org, _ = Org.objects.get_or_create(name=orgs[email])
        profile = UserProfile.objects.create(user=user, org=org)
        profile.group = person["role"]
        user.is_staff = user.is_superuser = person["role"] == "admin"
        user.save()
        teams = {}
        for name, is_admin in person.get("teams", {}).items():
            teams[name], _ = Team.objects.get_or_create(name=name, org=org)
            TeamMembership.objects.create(user_profile=profile, team=teams[name], is_admin=is_admin)
        for name in person.get("keys", []):
            add_key(user, name)
        for name, team in person.get("service_keys", {}).items():
            add_key(user, name, teams[team])
    print(f"    {email:<19} {person['role']:<10} {person['note']}")
