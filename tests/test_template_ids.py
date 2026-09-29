"""app.js runs top to bottom at load; one getElementById miss throws and skips everything after it
(including window.namTrainingHost, which the Continuous Gain tab needs to host training)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


def test_no_curly_quoted_attributes():
    assert not re.findall(r"\w+=[“”]", HTML)


def test_every_app_js_element_lookup_exists_in_template():
    ids = set(re.findall(r'\bid="([^"]+)"', HTML))
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    looked_up = set(re.findall(r'getElementById\("([^"$]+)"\)', js))
    assert sorted(looked_up - ids) == []
