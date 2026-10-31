"""Static checks for the match-ratings page in web/ (no browser needed).

They guard the contract between the three page files, the sample payload and the Python side:
files and credit, no external requests, the position coordinate table, the colour-band thresholds
shared with ``src/scale/bands.py``, the element ids ``app.js`` looks up, and text contrast in both themes.
"""
from __future__ import annotations

import json
import re

import pytest

from src.contract import MIN_MINUTES
from src.pipeline.export_web import index_entry, validate_payload
from src.scale.bands import rating_band
from tests.conftest import FIXTURES, ROOT

WEB = ROOT / "web"
HTML = (WEB / "index.html").read_text(encoding="utf-8")
JS = (WEB / "app.js").read_text(encoding="utf-8")
CSS = (WEB / "styles.css").read_text(encoding="utf-8")
STATSBOMB_URL = "https://github.com/statsbomb/open-data"


# ------------------------------------------------------------------ files and credit
def test_page_files_are_wired_together():
    assert re.search(r'<script[^>]+src="app\.js"', HTML)
    assert re.search(r'<link[^>]+rel="stylesheet"[^>]+href="styles\.css"', HTML)
    assert re.search(r'<meta[^>]+name="viewport"[^>]+width=device-width', HTML), "needed for phone widths"
    assert '<html lang="en">' in HTML and "<title>" in HTML


def test_statsbomb_credit_is_a_link_to_the_open_data_repo():
    assert re.search(rf'Data:\s*<a[^>]+href="{re.escape(STATSBOMB_URL)}"[^>]*>StatsBomb Open Data</a>', HTML)
    assert 'rel="noopener noreferrer"' in HTML


def test_page_makes_no_external_requests():
    allowed = {STATSBOMB_URL, "http://www.w3.org/2000/svg"}
    for name, text in (("index.html", HTML), ("app.js", JS), ("styles.css", CSS)):
        found = set(re.findall(r"https?://[^\s\"'<>)\\]+", text))
        assert found <= allowed, f"{name} refers to external URLs: {sorted(found - allowed)}"
    assert "@import" not in CSS
    assert not re.search(r"<(script|link)[^>]+(src|href)=\"(https?:)?//", HTML), "no CDN scripts or stylesheets"


def test_app_js_never_writes_markup_from_data():
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in JS, f"app.js uses {banned}: JSON strings must only reach the page via textContent"


def test_every_element_app_js_looks_up_exists_in_the_html():
    wanted = set(re.findall(r"\$\('#([\w-]+)", JS))
    assert wanted, "expected app.js to look up elements by id"
    present = set(re.findall(r'\bid="([\w-]+)"', HTML))
    assert wanted <= present, f"ids missing from index.html: {sorted(wanted - present)}"
    for needed in ("match-select", "method-toggle", "pitch", "markers", "players-table", "banner"):
        assert needed in present


def test_pitch_is_inline_svg_with_both_halves():
    svg = re.search(r'<svg id="pitch"[^>]*viewBox="([^"]+)"', HTML)
    assert svg, "the pitch must be inline SVG with a viewBox so it scales"
    assert HTML.count('class="goal"') == 2 and 'class="spot"' in HTML
    assert 'y1="420"' in HTML, "halfway line across the 840-unit pitch"


# ------------------------------------------------------------------ sample fallback
def load_sample():
    index = json.loads((WEB / "sample" / "index.json").read_text(encoding="utf-8"))
    entry = index[0]
    payload = json.loads((WEB / "sample" / entry["path"]).read_text(encoding="utf-8"))
    return index, entry, payload


def test_sample_index_has_one_entry_and_its_path_resolves_inside_sample():
    index, entry, payload = load_sample()
    assert len(index) == 1
    assert entry["path"] == "2015-16/1.json" and (WEB / "sample" / entry["path"]).is_file()
    assert entry == index_entry(payload)


def derive_sample(fixture: dict) -> dict:
    """web/sample/2015-16/1.json: the SYNTHETIC fixture plus synthetic minute_on / minute_off per player, so the
    bench list can show "on 69'". Starters are on at 0.0 and off at their minutes; substitutes come on at
    (match length - minutes) and stay to the final whistle."""
    length = max(p["minutes"] for p in fixture["players"])
    derived = json.loads(json.dumps(fixture))
    for i, p in enumerate(fixture["players"]):
        timed = {}
        for key, value in p.items():
            timed[key] = value
            if key == "minutes":
                timed["minute_on"], timed["minute_off"] = (0.0, p["minutes"]) if p["is_starter"] else (round(length - p["minutes"], 1), length)
        derived["players"][i] = timed
    return derived


def test_sample_payload_is_the_fixture_plus_synthetic_substitution_times():
    _, _, payload = load_sample()
    validate_payload(payload)
    fixture = json.loads((FIXTURES / "sample_match.json").read_text(encoding="utf-8"))
    assert "SYNTHETIC" in payload["note"]
    assert payload == derive_sample(fixture), (
        "web/sample/2015-16/1.json drifted from tests/fixtures/sample_match.json. Regenerate it with "
        "json.dumps(derive_sample(fixture), indent=2) (see derive_sample in this file); index.json comes from "
        "src.pipeline.export_web.index_entry.")
    subs = [p for p in payload["players"] if not p["is_starter"]]
    assert subs and all(abs(p["minute_on"] + p["minutes"] - p["minute_off"]) < 1e-6 for p in subs)
    assert all(p["minute_on"] == 0.0 for p in payload["players"] if p["is_starter"])


def test_app_js_falls_back_to_the_sample_and_says_so():
    assert "'data/index.json'" in JS and "'sample/index.json'" in JS
    assert "synthetic sample data" in JS.lower()
    assert 'id="banner"' in HTML and re.search(r'id="banner"[^>]*\bhidden\b', HTML), "banner is hidden until needed"


# ------------------------------------------------------------------ colour bands and constants
def js_list(name: str) -> list:
    body = re.search(rf"const {name} = \[([^\]]+)\]", JS).group(1)
    return [item.strip().strip("'\"") for item in body.split(",")]


def test_js_band_thresholds_match_src_scale_bands():
    edges = [float(x) for x in js_list("BAND_EDGES")]
    names = js_list("BAND_NAMES")
    assert len(names) == len(edges) + 1
    for i, edge in enumerate(edges):
        assert rating_band(edge - 0.1) == names[i], f"just below {edge}"
        assert rating_band(edge) == names[i + 1], f"at {edge}"
    assert rating_band(10.0) == names[-1] and rating_band(None) == "unrated"
    assert f"band-{names[0]}" in CSS and "band-unrated" in CSS


def test_js_min_minutes_matches_the_contract():
    assert float(re.search(r"const MIN_MINUTES = ([\d.]+);", JS).group(1)) == MIN_MINUTES


# ------------------------------------------------------------------ position coordinate table
def position_layout() -> dict:
    block = re.search(r"const POSITION_LAYOUT = \{(.*?)\n  \};", JS, re.S).group(1)
    return {int(k): (float(u), float(d)) for k, u, d in
            re.findall(r"(\d+):\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", block)}


def test_position_layout_covers_statsbomb_ids_1_to_25_inside_the_half():
    layout = position_layout()
    assert sorted(layout) == list(range(1, 26))
    assert all(0.0 < u < 1.0 and 0.0 < d < 1.0 for u, d in layout.values())


def test_position_layout_puts_right_on_the_right_when_attacking_upwards():
    u = {k: v[0] for k, v in position_layout().items()}
    for right, left in ((2, 6), (3, 5), (7, 8), (9, 11), (12, 16), (13, 15), (17, 21), (18, 20), (22, 24)):
        assert u[right] > u[left], f"position {right} must be right of {left}"
        assert u[right] + u[left] == pytest.approx(1.0), f"{right}/{left} should mirror each other"
    for centre in (1, 4, 10, 14, 19, 23, 25):
        assert u[centre] == 0.5


def test_position_layout_depths_follow_the_lines_of_the_team():
    d = {k: v[1] for k, v in position_layout().items()}
    assert d[1] == min(d.values()), "the goalkeeper is on the goal line"
    back_line = [d[i] for i in (2, 3, 4, 5, 6)]
    midfield = [d[i] for i in (9, 10, 11, 12, 13, 14, 15, 16)]
    attacking = [d[i] for i in (17, 18, 19, 20, 21)]
    forwards = [d[i] for i in (22, 23, 24)]
    assert d[1] < min(back_line)
    assert max(back_line) < min(midfield)
    assert max(midfield) < min(attacking) <= max(attacking) < min(forwards)
    assert d[7] > max(back_line) and d[8] > max(back_line), "wing backs push up beyond the back line"
    assert d[10] < d[14], "defensive midfielders sit deeper than central midfielders"
    assert d[19] < d[23] and d[25] < d[23], "the secondary striker drops behind the centre forward"
    assert d[17] > d[14] and d[21] > d[14], "wingers play higher than central midfielders"


def test_every_position_has_an_abbreviation():
    abbr = re.search(r"const POSITION_ABBR = \{(.*?)\};", JS, re.S).group(1)
    assert sorted(int(k) for k in re.findall(r"(\d+):", abbr)) == list(range(1, 26))


# ------------------------------------------------------------------ contrast in both themes
def block_after(text: str, marker: str) -> str:
    """Contents of the {...} block that starts at the first '{' after ``marker`` (brace-balanced)."""
    start = text.index("{", text.index(marker))
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i]
    raise AssertionError(f"unbalanced braces after {marker!r}")


def hex_tokens(block: str) -> dict:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def theme_tokens() -> dict:
    light = hex_tokens(block_after(CSS, ":root {"))
    dark = {**light, **hex_tokens(block_after(block_after(CSS, "@media (prefers-color-scheme: dark)"), ":root"))}
    return {"light": light, "dark": dark}


def luminance(color: str) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


TEXT_PAIRS = [  # (foreground token, background token): all are text, so WCAG AA 4.5:1
    ("band-red-ink", "band-red"), ("band-orange-ink", "band-orange"), ("band-green-ink", "band-green"),
    ("band-blue-ink", "band-blue"), ("band-unrated-ink", "band-unrated"),
    ("ink", "surface"), ("ink-2", "surface"), ("ink-3", "surface"), ("ink", "bg"), ("ink-3", "bg"),
    ("ink-2", "surface-2"), ("ink-3", "surface-2"), ("ink", "hl"),
    ("seg-on-ink", "seg-on-bg"), ("banner-ink", "banner-bg"), ("notice-ink", "notice-bg"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS, ids=[f"{a}-on-{b}" for a, b in TEXT_PAIRS])
def test_text_meets_wcag_aa_contrast(theme, fg, bg):
    tokens = theme_tokens()[theme]
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= 4.5, f"{fg} {tokens[fg]} on {bg} {tokens[bg]} is {ratio:.2f}:1 in the {theme} theme"


def test_white_pitch_labels_stay_readable_on_the_grass_in_both_themes():
    for theme, tokens in theme_tokens().items():
        for grass in ("pitch-a", "pitch-b"):
            assert contrast("#ffffff", tokens[grass]) >= 4.0, f"{grass} in the {theme} theme"


def test_badge_colours_do_not_change_between_themes():
    tokens = theme_tokens()
    for band in ("red", "orange", "green", "blue"):
        assert tokens["light"][f"band-{band}"] == tokens["dark"][f"band-{band}"]


def test_css_supports_both_os_themes_and_phone_widths():
    assert "color-scheme: light dark" in CSS
    assert "@media (prefers-color-scheme: dark)" in CSS
    assert re.search(r"@media \(max-width: 6\d\dpx\)", CSS), "phone breakpoint"
    assert re.search(r"@media \(min-width: \d+px\)", CSS), "wide-screen layout"
    assert "prefers-reduced-motion" in CSS


# ------------------------------------------------------------------ goalkeepers in VAEP mode show their baseline rating
GK_NOTE = "Goalkeepers show their baseline rating: VAEP does not value shot-stopping reliably."


def js_function(name: str) -> str:
    start = JS.index(f"function {name}(")
    return JS[start:JS.index("\n  }\n", start)]


def test_vaep_goalkeeper_note_is_in_the_page_and_only_shown_for_vaep():
    assert GK_NOTE in JS
    assert re.search(r'id="gk-note"[^>]*\bhidden\b', HTML), "the note starts hidden"
    assert "ui.gkNote.hidden = state.method !== 'vaep'" in JS


def test_goalkeeper_substitution_logic_uses_the_baseline_rating():
    body = js_function("displayed")
    assert "method === 'vaep'" in body and "position_group === 'GK'" in body
    assert "ratingOf(player, 'baseline')" in body
    assert "swapped: rating !== null" in body, "a keeper with no baseline rating stays plain unrated"
    assert "source: 'baseline'" in body


def test_pitch_table_bench_sort_and_tooltips_all_use_the_displayed_value():
    assert "displayed(p, method)" in js_function("renderPitch") + js_function("renderSubs") + js_function("renderTable")
    for name in ("renderPitch", "renderSubs", "renderTable", "compareForTable"):
        assert "ratingOf(" not in js_function(name).replace("displayed(", ""), f"{name} must not read ratingOf directly"
    assert "displayed(a, method).rating" in js_function("compareForTable"), "table sorts by the displayed value"
    assert "displayed(player, method)" in js_function("tooltipFor"), "tooltips show the baseline value for keepers"
    assert "is-baseline" in js_function("badge") and "btag" in js_function("badge")
    assert "btag-text" in js_function("renderPitch"), "pitch badge carries the B tag"
    assert ".badge.is-baseline" in CSS and ".marker.is-baseline" in CSS
