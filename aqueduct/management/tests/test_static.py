import re
from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.template.utils import get_app_template_dirs
from django.test import SimpleTestCase

STATIC_TAG = re.compile(r"""\{%\s*static\s+["']([^"']+)["']""")


class StaticReferencesTest(SimpleTestCase):
    def test_every_static_path_in_our_templates_exists(self):
        base = Path(settings.BASE_DIR)
        dirs = [base / "templates"] + [
            d for d in get_app_template_dirs("templates") if d.is_relative_to(base)
        ]
        references = {
            (template.relative_to(base), path)
            for directory in dirs
            for template in directory.rglob("*.html")
            for path in STATIC_TAG.findall(template.read_text())
        }
        self.assertTrue(references)
        missing = sorted(ref for ref in references if finders.find(ref[1]) is None)
        self.assertEqual(missing, [])
