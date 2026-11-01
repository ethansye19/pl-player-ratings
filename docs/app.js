/* Premier League match ratings page.
 *
 * Vanilla JS, no build step, no network access beyond the static JSON next to this page:
 * data/index.json (real export) or, when that is missing, sample/index.json (SYNTHETIC).
 * Payload schema: CLAUDE.md "Web payload". Strings from the JSON only ever reach the page
 * through textContent / setAttribute, so they can never be parsed as markup.
 */
(function () {
  'use strict';

  /* ================================================================ configuration */
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const DATA_INDEX_URL = 'data/index.json';
  const SAMPLE_INDEX_URL = 'sample/index.json';
  const STORAGE_KEY = 'matchRatings.method';

  const MIN_MINUTES = 10;                          // src/contract.py: no rating under this
  const BAND_EDGES = [6.0, 7.0, 8.0];              // src/scale/bands.py: red < 6 <= orange < 7 <= green < 8 <= blue
  const BAND_NAMES = ['red', 'orange', 'green', 'blue'];

  const METHOD_LABELS = { baseline: 'Baseline', vaep: 'VAEP' };
  const METHOD_BLURBS = {
    baseline: 'hand-weighted match stats, with separate weights per position group',
    vaep: 'the value of every on-ball action (socceraction VAEP), summed per player'
  };

  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  /* ---------------------------------------------------------------- pitch geometry
   * SVG user units: the pitch is 544 x 840 (68 m x 105 m at 8 units per metre); index.html draws
   * the markings. The home team is the bottom half (attacking up), away the top half (attacking down). */
  const PITCH_W = 544;
  const PITCH_H = 840;
  const HALF_H = PITCH_H / 2;
  const MARKER_BOX = { w: 100, h: 60 };            // footprint used to keep neighbours apart
  const MARKER_EDGE = { x: 50, near: 24, far: 22 }; // min distance of a marker centre from touchline / own goal line / halfway line
  const MARKER_LABEL_MAX = 148;                    // absolute cap for a name label (SVG units)

  /* StatsBomb position_id -> [u, d] in the team's own frame:
   *   u = sideways, 0 = the team's own LEFT touchline, 1 = its own RIGHT (as the team faces the opposition goal);
   *   d = depth, 0 = own goal line, 1 = halfway line.
   * Home is drawn with u -> left-to-right and d -> upwards, so a right back (u high) is on the viewer's right;
   * away is mirrored (rotated 180 degrees) so it faces the home side. Ids 1-25 are all covered (CLAUDE.md,
   * "Position groups"); anything else falls back on its position_group depth. */
  const POSITION_LAYOUT = {
    1: [0.50, 0.10],   // Goalkeeper (inside the 6-yard box, label stays on the pitch)
    2: [0.88, 0.27],   // Right Back
    3: [0.64, 0.24],   // Right Center Back
    4: [0.50, 0.24],   // Center Back
    5: [0.36, 0.24],   // Left Center Back
    6: [0.12, 0.27],   // Left Back
    7: [0.90, 0.42],   // Right Wing Back
    8: [0.10, 0.42],   // Left Wing Back
    9: [0.64, 0.43],   // Right Defensive Midfield
    10: [0.50, 0.43],  // Center Defensive Midfield
    11: [0.36, 0.43],  // Left Defensive Midfield
    12: [0.88, 0.57],  // Right Midfield
    13: [0.64, 0.57],  // Right Center Midfield
    14: [0.50, 0.57],  // Center Midfield
    15: [0.36, 0.57],  // Left Center Midfield
    16: [0.12, 0.57],  // Left Midfield
    17: [0.88, 0.74],  // Right Wing
    18: [0.68, 0.68],  // Right Attacking Midfield
    19: [0.50, 0.68],  // Center Attacking Midfield
    20: [0.32, 0.68],  // Left Attacking Midfield
    21: [0.12, 0.74],  // Left Wing
    22: [0.64, 0.86],  // Right Center Forward
    23: [0.50, 0.88],  // Center Forward
    24: [0.36, 0.86],  // Left Center Forward
    25: [0.50, 0.78]   // Secondary Striker
  };
  const GROUP_DEPTH = { GK: 0.10, DEF: 0.26, MID: 0.57, FWD: 0.86 };

  const POSITION_ABBR = {
    1: 'GK', 2: 'RB', 3: 'RCB', 4: 'CB', 5: 'LCB', 6: 'LB', 7: 'RWB', 8: 'LWB', 9: 'RDM', 10: 'CDM',
    11: 'LDM', 12: 'RM', 13: 'RCM', 14: 'CM', 15: 'LCM', 16: 'LM', 17: 'RW', 18: 'RAM', 19: 'CAM',
    20: 'LAM', 21: 'LW', 22: 'RCF', 23: 'CF', 24: 'LCF', 25: 'SS'
  };

  const NAME_PARTICLES = ['van', 'von', 'de', 'di', 'da', 'do', 'dos', 'das', 'del', 'della', 'der', 'den',
    'ter', 'ten', 'la', 'le', 'el', 'al', 'bin', 'ibn'];
  const NAME_SUFFIXES = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv'];

  /* ================================================================ state */
  const state = {
    index: [],            // normalised index entries, sorted by season / matchweek / date
    indexUrl: DATA_INDEX_URL,
    sample: false,        // true when showing the synthetic fallback
    filters: { season: '', team: '' },
    matchId: null,
    payload: null,
    placed: [],           // pitch positions of the current match's starters
    method: 'baseline',
    token: 0,             // guards against out-of-order fetches
    cache: new Map(),
    highlighted: null
  };
  const ui = {};          // cached DOM nodes, filled in by init()

  /* ================================================================ small helpers */
  function $(selector, root) { return (root || document).querySelector(selector); }
  function $$(selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); }

  function applyProps(node, props) {
    if (!props) return;
    Object.keys(props).forEach(function (key) {
      const value = props[key];
      if (value === null || value === undefined || value === false) return;
      if (key === 'text') node.textContent = String(value);
      else if (key.slice(0, 2) === 'on' && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : String(value));
    });
  }
  function appendAll(node, children) {
    children.forEach(function (child) {
      if (child === null || child === undefined || child === false) return;
      if (Array.isArray(child)) appendAll(node, child);
      else node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    });
  }
  function h(tag, props) {
    const node = document.createElement(tag);
    applyProps(node, props);
    appendAll(node, Array.prototype.slice.call(arguments, 2));
    return node;
  }
  function s(tag, props) {
    const node = document.createElementNS(SVG_NS, tag);
    applyProps(node, props);
    appendAll(node, Array.prototype.slice.call(arguments, 2));
    return node;
  }
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }
  function cssEscape(value) {
    return window.CSS && typeof window.CSS.escape === 'function' ? window.CSS.escape(String(value)) : String(value).replace(/[^\w-]/g, '\\$&');
  }
  function round1(value) { return Math.round(value * 10) / 10; }

  function toNumber(value) {
    if (value === null || value === undefined || typeof value === 'boolean') return null;
    if (typeof value === 'string' && value.trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  /* ================================================================ domain helpers */
  /* Same thresholds as src/scale/bands.py rating_band(). */
  function ratingBand(rating) {
    const value = toNumber(rating);
    if (value === null) return 'unrated';
    for (let i = 0; i < BAND_EDGES.length; i++) {
      if (value < BAND_EDGES[i]) return BAND_NAMES[i];
    }
    return BAND_NAMES[BAND_NAMES.length - 1];
  }

  function ratingOf(player, method) {
    return player && player.ratings ? toNumber(player.ratings[method]) : null;
  }
  /* What the page shows for a player under the selected method. VAEP does not rate goalkeepers reliably, so in
   * VAEP mode a goalkeeper shows his BASELINE rating (null when he has none: shown as unrated). `source` is the
   * method whose number is displayed; `swapped` is true when that differs from the selected method. */
  const GK_NOTE = 'Goalkeepers show their baseline rating: VAEP does not value shot-stopping reliably.';
  function displayed(player, method) {
    if (method === 'vaep' && player && player.position_group === 'GK') {
      const rating = ratingOf(player, 'baseline');
      return { rating: rating, source: 'baseline', swapped: rating !== null };
    }
    return { rating: ratingOf(player, method), source: method, swapped: false };
  }
  function rawScoreOf(player, method) {
    return player && player.raw_scores ? toNumber(player.raw_scores[method]) : null;
  }
  function methodLabel(method) {
    return METHOD_LABELS[method] || (String(method).charAt(0).toUpperCase() + String(method).slice(1));
  }

  function fmtRating(rating) { return rating === null ? '–' : rating.toFixed(1); }
  function fmtMinutes(minutes) {
    const v = toNumber(minutes);
    if (v === null) return '–';
    return v < MIN_MINUTES ? v.toFixed(1) : String(Math.round(v));
  }
  function fmtClock(minute) { return Math.round(minute) + '′'; }
  function fmtScore(score) { const v = toNumber(score); return v === null ? '–' : String(v); }
  function fmtFormation(value) {
    if (value === null || value === undefined || String(value).trim() === '') return '–';
    const text = String(value).trim();
    return /^\d+$/.test(text) ? text.split('').join('-') : text;   // "4231" -> "4-2-3-1"
  }

  function parseDay(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    if (!m) return null;
    const date = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
    return Number.isNaN(date.getTime()) ? null : { year: +m[1], month: +m[2] - 1, day: +m[3], dow: date.getUTCDay() };
  }
  function fmtDate(iso, long) {
    const d = parseDay(iso);
    if (!d) return iso ? String(iso) : '';
    const short = d.day + ' ' + MONTHS[d.month];
    return long ? WEEKDAYS[d.dow] + ' ' + short + ' ' + d.year : short;
  }
  function fmtDateRange(fromIso, toIso) {
    const a = parseDay(fromIso), b = parseDay(toIso);
    if (!a || !b) return '';
    if (a.year === b.year && a.month === b.month) {
      return a.day === b.day ? a.day + ' ' + MONTHS[a.month] : a.day + '–' + b.day + ' ' + MONTHS[a.month];
    }
    return a.day + ' ' + MONTHS[a.month] + ' – ' + b.day + ' ' + MONTHS[b.month];
  }

  /* "Matt Ritchie" -> "Ritchie", "Virgil van Dijk" -> "van Dijk", "Neymar Jr" -> "Neymar". */
  function shortName(fullName) {
    const parts = String(fullName === null || fullName === undefined ? '' : fullName).trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) return '?';
    if (parts.length === 1) return parts[0];
    let end = parts.length;
    while (end > 1 && NAME_SUFFIXES.indexOf(parts[end - 1].toLowerCase()) !== -1) end--;
    let start = end - 1;
    if (/^\d+$/.test(parts[start]) && start > 0) start--;          // "Synthetic Home 4" -> "Home 4"
    while (start > 1 && NAME_PARTICLES.indexOf(parts[start - 1].toLowerCase()) !== -1) start--;
    return parts.slice(start, end).join(' ');
  }

  function positionAbbr(player) {
    return POSITION_ABBR[player.position_id] || player.position_group || '';
  }

  /* ================================================================ pitch layout */
  /* Push apart any two markers whose footprints overlap (colliding position_ids do occur in real
   * lineups: a player who moved spends most minutes elsewhere, and group-only lineups put a whole
   * line on one spot). Prefers sideways moves, so a line stays a line. */
  function resolveOverlaps(nodes, boxW, boxH, bounds) {
    function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
    for (let pass = 0; pass < 80; pass++) {
      let moved = false;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const overlapX = boxW - Math.abs(dx), overlapY = boxH - Math.abs(dy);
          if (overlapX <= 0 || overlapY <= 0) continue;
          moved = true;
          // players level with each other are always separated sideways; only a marker in a different row moves up or down
          if (Math.abs(dy) < boxH * 0.45 || overlapX / boxW <= overlapY / boxH) {
            const dir = dx > 0 ? 1 : dx < 0 ? -1 : 1;
            const push = overlapX / 2 + 0.5;
            a.x -= dir * push; b.x += dir * push;
          } else {
            const dir = dy > 0 ? 1 : dy < 0 ? -1 : 1;
            const push = overlapY / 2 + 0.5;
            a.y -= dir * push; b.y += dir * push;
          }
        }
      }
      nodes.forEach(function (n) {
        n.x = clamp(n.x, bounds.minX, bounds.maxX);
        n.y = clamp(n.y, bounds.minY, bounds.maxY);
      });
      if (!moved) break;
    }
  }

  /* Screen positions for one side's starters. Returns [{player, x, y}] in SVG units. */
  function layoutTeam(starters, side) {
    const nodes = starters.map(function (player) {
      const spec = POSITION_LAYOUT[player.position_id] || [0.5, GROUP_DEPTH[player.position_group] || 0.57];
      return { player: player, x: spec[0] * PITCH_W, y: spec[1] * HALF_H };
    });
    resolveOverlaps(nodes, MARKER_BOX.w, MARKER_BOX.h, {
      minX: MARKER_EDGE.x, maxX: PITCH_W - MARKER_EDGE.x,
      minY: MARKER_EDGE.near, maxY: HALF_H - MARKER_EDGE.far
    });
    return nodes.map(function (n) {
      return side === 'home'
        ? { player: n.player, x: round1(n.x), y: round1(PITCH_H - n.y) }      // attacking up: own goal at the bottom
        : { player: n.player, x: round1(PITCH_W - n.x), y: round1(n.y) };     // attacking down, mirrored to face home
    });
  }

  /* Widest label (SVG units) each marker may use: the room to its nearest same-row neighbour, within the
   * pitch edges. Never narrower than the marker footprint, so 5-across rows stay tidy and wide rows keep long surnames whole. */
  function assignLabelBudgets(placed) {
    placed.forEach(function (a) {
      let room = MARKER_LABEL_MAX;
      placed.forEach(function (b) {
        if (a !== b && Math.abs(a.y - b.y) < 46) room = Math.min(room, Math.abs(a.x - b.x) - 6);
      });
      const edge = 2 * (Math.min(a.x, PITCH_W - a.x) + 18);
      a.labelMax = Math.max(MARKER_BOX.w - 2, Math.min(room, edge));
    });
    return placed;
  }

  function layoutMatch(payload) {
    let placed = [];
    ['home', 'away'].forEach(function (side) {
      const starters = payload.players.filter(function (p) { return p.side === side && p.is_starter; });
      placed = placed.concat(layoutTeam(starters, side));
    });
    return assignLabelBudgets(placed);
  }

  /* Shrink then truncate an SVG <text> so it fits maxWidth (user units). */
  function fitText(node, maxWidth) {
    if (typeof node.getComputedTextLength !== 'function') return;
    let length;
    try { length = node.getComputedTextLength(); } catch (err) { return; }
    if (!length || length <= maxWidth) return;
    const baseSize = parseFloat(node.getAttribute('data-size')) || 20;
    node.style.fontSize = (baseSize * Math.max(0.8, 0.97 * maxWidth / length)).toFixed(1) + 'px';
    let text = node.textContent;
    try {
      length = node.getComputedTextLength();
      while (length > maxWidth && text.length > 2) {
        text = text.slice(0, -1);
        node.textContent = text.replace(/[\s-]+$/, '') + '…';
        length = node.getComputedTextLength();
      }
    } catch (err) { /* measuring unavailable: leave the text as is */ }
  }

  /* ================================================================ rendering */
  function tooltipFor(player, payload, method) {
    const bits = [player.player_name, '#' + player.jersey_number, player.position || positionAbbr(player), fmtMinutes(player.minutes) + ' min'];
    if (method === 'vaep' && player.position_group === 'GK') {
      const shown = displayed(player, method);
      return bits.concat(shown.rating === null ? 'Baseline unrated (VAEP is not used for goalkeepers)'
        : 'Baseline ' + fmtRating(shown.rating) + ' (shown instead of VAEP for goalkeepers)').join(' \u00b7 ');
    }
    const ratings = payload.methods.map(function (m) {
      const rating = ratingOf(player, m);
      return methodLabel(m) + ' ' + (rating === null ? 'unrated' : fmtRating(rating));
    });
    return bits.concat(ratings.join(' / ')).join(' · ');
  }

  function badge(rating, title, swapped) {
    const band = ratingBand(rating);
    return h('span', { class: 'badge band-' + band + (swapped ? ' is-baseline' : ''), title: title },
      rating === null
        ? [h('span', { 'aria-hidden': 'true', text: '–' }), h('span', { class: 'visually-hidden', text: 'unrated' })]
        : [fmtRating(rating), swapped ? h('span', { class: 'btag', 'aria-hidden': 'true', text: 'B' }) : null,
           swapped ? h('span', { class: 'visually-hidden', text: ' (baseline rating)' }) : null]);
  }

  function renderHeader(payload) {
    const head = clear(ui.head);
    const meta = [[payload.competition, payload.season].filter(Boolean).join(' '),
      toNumber(payload.matchweek) !== null ? 'Matchweek ' + payload.matchweek : '',
      fmtDate(payload.match_date, true)].filter(Boolean).join(' · ');
    function teamBlock(team, side) {
      return h('div', { class: 'team team-' + side },
        h('span', { class: 'team-name', text: team.team || (side === 'home' ? 'Home' : 'Away') }),
        h('span', { class: 'tags' },
          h('span', { class: 'tag', text: side === 'home' ? 'Home' : 'Away' }),
          h('span', { class: 'tag', title: 'Formation', text: fmtFormation(team.formation) })));
    }
    head.appendChild(h('p', { class: 'meta', text: meta }));
    head.appendChild(h('div', { class: 'teams' },
      teamBlock(payload.home, 'home'),
      h('div', { class: 'score', role: 'img',
        'aria-label': 'Score: ' + payload.home.team + ' ' + fmtScore(payload.home.score) + ', ' + payload.away.team + ' ' + fmtScore(payload.away.score) },
      h('span', { text: fmtScore(payload.home.score) }), h('span', { class: 'dash', 'aria-hidden': 'true', text: '–' }), h('span', { text: fmtScore(payload.away.score) })),
      teamBlock(payload.away, 'away')));
    document.title = payload.home.team + ' ' + fmtScore(payload.home.score) + '–' + fmtScore(payload.away.score) + ' ' + payload.away.team + ' · Match ratings';
  }

  function renderMethodToggle(payload) {
    const fieldset = ui.toggle;
    const methods = payload.methods;
    // add radios for any method the page does not know yet, so every payload method can be selected
    methods.forEach(function (m) {
      if (!$('input[name="method"][value="' + cssEscape(m) + '"]', fieldset)) {
        fieldset.appendChild(h('label', { class: 'seg' }, h('input', { type: 'radio', name: 'method', value: m }), h('span', { text: methodLabel(m) })));
      }
    });
    $$('input[name="method"]', fieldset).forEach(function (input) {
      input.disabled = methods.indexOf(input.value) === -1;
      input.checked = input.value === state.method;
    });
    const single = methods.length < 2;
    fieldset.disabled = single;
    fieldset.title = single ? 'Only one rating method is available for this match' : '';
  }

  function renderPitch(payload, method) {
    const layer = clear(ui.markers);
    state.placed.forEach(function (item) {
      const p = item.player;
      const shown = displayed(p, method);
      const rating = shown.rating;
      layer.appendChild(s('g', {
        class: 'marker band-' + ratingBand(rating) + ' side-' + p.side + (shown.swapped ? ' is-baseline' : ''),
        transform: 'translate(' + item.x + ' ' + item.y + ')',
        tabindex: '0', role: 'img', 'data-pid': p.player_id,
        'aria-label': p.player_name + ', ' + (p.position || positionAbbr(p)) + ', ' + methodLabel(shown.source) + ' ' + (rating === null ? 'unrated' : 'rating ' + fmtRating(rating))
      },
      s('title', null, tooltipFor(p, payload, method)),
      s('rect', { class: 'pill', x: -29, y: -17, width: 58, height: 34, rx: 9 }),
      s('text', { class: 'rtext', x: 0, y: 0, dy: '.35em', text: fmtRating(rating) }),
      shown.swapped ? s('circle', { class: 'btag-bg', cx: 27, cy: -17, r: 10 }) : null,
      shown.swapped ? s('text', { class: 'btag-text', x: 27, y: -17, dy: '.35em', text: 'B' }) : null,
      s('text', { class: 'pname', x: 0, y: 40, 'data-size': 20, text: shortName(p.player_name) })));
    });
    $$('.marker', layer).forEach(function (marker, i) {
      fitText($('.pname', marker), state.placed[i].labelMax || MARKER_BOX.w - 2);
    });

    ui.noteAway.textContent = '▼ Away · ' + (payload.away.team || '');
    ui.noteHome.textContent = '▲ Home · ' + (payload.home.team || '');
    ui.noteAwayForm.textContent = fmtFormation(payload.away.formation);
    ui.noteHomeForm.textContent = fmtFormation(payload.home.formation);
    ui.noteAway.setAttribute('data-size', '16');
    ui.noteHome.setAttribute('data-size', '16');
    ui.noteAway.style.fontSize = ui.noteHome.style.fontSize = '';
    fitText(ui.noteAway, 222);
    fitText(ui.noteHome, 222);
  }

  function matchEndMinute(players) {
    let end = 0;
    players.forEach(function (p) {
      const off = toNumber(p.minute_off);
      const v = off !== null ? off : toNumber(p.minutes);
      if (v !== null && v > end) end = v;
    });
    return end;
  }

  function renderSubs(payload, method) {
    const end = matchEndMinute(payload.players);
    [['home', ui.subsHome], ['away', ui.subsAway]].forEach(function (pair) {
      const side = pair[0], box = pair[1];
      const team = payload[side];
      const title = $('h3', box);
      clear(title);
      title.appendChild(h('span', { class: 'side-dot ' + side, title: side === 'home' ? 'Home' : 'Away' }));
      title.appendChild(document.createTextNode((team.team || (side === 'home' ? 'Home' : 'Away')) + ' '));
      title.appendChild(h('span', { class: 'subs-team', text: 'substitutes' }));
      const list = clear($('.sub-list', box));
      const subs = payload.players.filter(function (p) { return p.side === side && !p.is_starter; });
      subs.sort(function (a, b) {
        const ao = toNumber(a.minute_on), bo = toNumber(b.minute_on);
        if (ao !== null && bo !== null && ao !== bo) return ao - bo;
        if (ao !== null && bo === null) return -1;
        if (ao === null && bo !== null) return 1;
        return (toNumber(a.jersey_number) || 0) - (toNumber(b.jersey_number) || 0);
      });
      if (subs.length === 0) {
        list.appendChild(h('li', { class: 'empty', text: 'No substitutes came on.' }));
        return;
      }
      subs.forEach(function (p) {
        const on = toNumber(p.minute_on), off = toNumber(p.minute_off);
        const leftEarly = on !== null && off !== null && off < end - 0.5;
        list.appendChild(h('li', { class: 'sub', 'data-pid': p.player_id },
          badge(displayed(p, method).rating, tooltipFor(p, payload, method), displayed(p, method).swapped),
          h('span', { class: 'who' },
            h('span', { class: 'who-name', title: p.player_name, text: '#' + p.jersey_number + ' ' + shortName(p.player_name) }),
            h('span', { class: 'who-pos', text: positionAbbr(p) + (p.position ? ' · ' + p.position : '') })),
          h('span', { class: 'when' },
            on !== null ? 'on ' + fmtClock(on) : fmtMinutes(p.minutes) + ' min',
            h('small', { text: (leftEarly ? 'off ' + fmtClock(off) + ' · ' : '') + (on !== null ? fmtMinutes(p.minutes) + ' min' : '') }))));
      });
    });
  }

  function compareForTable(method) {
    return function (a, b) {
      const ra = displayed(a, method).rating, rb = displayed(b, method).rating;     // sort by what is shown
      if (ra !== null && rb !== null) { if (ra !== rb) return rb - ra; }
      else if (ra !== null) return -1;
      else if (rb !== null) return 1;
      const ma = toNumber(a.minutes) || 0, mb = toNumber(b.minutes) || 0;
      if (ma !== mb) return mb - ma;                       // ties: more minutes first (never raw scores: only comparable within a position group)
      return String(a.player_name).localeCompare(String(b.player_name));
    };
  }

  function renderTable(payload, method) {
    const rows = payload.players.slice().sort(compareForTable(method));
    const body = clear(ui.tableBody);
    let rank = 0;
    rows.forEach(function (p) {
      const shown = displayed(p, method);
      const rating = shown.rating;
      const raw = rawScoreOf(p, shown.source);
      const title = tooltipFor(p, payload, method) + (raw === null ? '' : ' · ' + methodLabel(shown.source) + ' raw score ' + raw);
      body.appendChild(h('tr', { 'data-pid': p.player_id },
        h('td', { class: 'c-rank', text: rating === null ? '–' : String(++rank) }),
        h('td', { class: 'c-name' },
          h('span', { class: 'side-dot ' + p.side, title: p.side === 'home' ? 'Home' : 'Away' }),
          h('span', { class: 'pnum', text: '#' + p.jersey_number }),
          h('span', { class: 'pn', text: p.player_name }),
          h('span', { class: 'pteam', text: p.team + (p.is_starter ? '' : ' · sub') })),
        h('td', { class: 'c-team', text: p.team + (p.is_starter ? '' : ' (sub)') }),
        h('td', { class: 'c-pos' }, h('span', { class: 'grp', title: p.position || '', text: p.position_group || '' })),
        h('td', { class: 'c-min', text: fmtMinutes(p.minutes) }),
        h('td', { class: 'c-rating' }, badge(rating, title, shown.swapped))));
    });
    const label = methodLabel(method);
    ui.ratingHeader.textContent = 'Rating ↓';
    ui.sortNote.textContent = 'Sorted by ' + label + ' rating, highest first. Unrated players last.' + (method === 'vaep' ? ' Goalkeepers are ranked by their baseline rating.' : '');
    ui.tableCaption.textContent = 'Players in ' + payload.home.team + ' v ' + payload.away.team + ', sorted by ' + label + ' rating';
  }

  /* Footer credit from payload.credits, written "Data: Name (url)": the name becomes the link. A StatsBomb credit
   * without a URL keeps the built-in link to the open-data repository. */
  function renderCredit(payload) {
    const text = typeof payload.credits === 'string' ? payload.credits.trim() : '';
    if (!text) return;
    const url = /https?:\/\/[^\s)]+/.exec(text);
    const credit = clear(ui.credit);
    if (!url && /statsbomb/i.test(text)) {
      ui.creditDefault.forEach(function (node) { credit.appendChild(node.cloneNode(true)); });
      return;
    }
    if (!url) { credit.textContent = text; return; }
    const link = function (label) {
      return h('a', { href: url[0], target: '_blank', rel: 'noopener noreferrer', text: label });
    };
    const before = text.slice(0, url.index).replace(/\s*\(\s*$/, '');
    const m = /^(.*?:\s*)(.+)$/.exec(before);                  // "Data: " + "StatsBomb Open Data"
    if (m) credit.appendChild(document.createTextNode(m[1]));
    credit.appendChild(link(m ? m[2] : (before || url[0])));
  }

  function renderAll() {
    const payload = state.payload;
    const label = methodLabel(state.method);
    renderHeader(payload);
    renderMethodToggle(payload);
    renderPitch(payload, state.method);
    renderSubs(payload, state.method);
    renderTable(payload, state.method);
    renderCredit(payload);
    ui.methodNote.textContent = label + ': ' + (METHOD_BLURBS[state.method] || 'rating method') + '.';
    ui.lineupsTitle.textContent = 'Lineups · ' + label + ' ratings';
    ui.gkNote.textContent = GK_NOTE;
    ui.gkNote.hidden = state.method !== 'vaep';
    setHighlight(null);
  }

  /* ================================================================ highlight (pitch <-> table <-> bench) */
  function setHighlight(pid) {
    const next = pid === null || pid === undefined ? null : String(pid);
    if (next === state.highlighted) return;
    $$('.is-hl').forEach(function (node) { node.classList.remove('is-hl'); });
    state.highlighted = next;
    if (next !== null) {
      $$('[data-pid="' + cssEscape(next) + '"]').forEach(function (node) { node.classList.add('is-hl'); });
    }
  }
  function onPointerOver(event) {
    const target = event.target && event.target.closest ? event.target.closest('[data-pid]') : null;
    setHighlight(target ? target.getAttribute('data-pid') : null);
  }

  /* ================================================================ loading */
  class LoadError extends Error {
    constructor(message, status) { super(message); this.name = 'LoadError'; this.status = status || null; }
  }

  async function fetchJson(url, options) {
    let response;
    try { response = await fetch(url, options); }
    catch (err) { throw new LoadError('could not fetch ' + url + ' (network error)'); }
    if (!response.ok) throw new LoadError(url + ' returned HTTP ' + response.status, response.status);
    try { return await response.json(); }
    catch (err) { throw new LoadError(url + ' is not valid JSON'); }
  }

  function byWeekThenDate(a, b) {
    if (a.season !== b.season) return a.season < b.season ? -1 : 1;
    const wa = a.matchweek === null ? Infinity : a.matchweek, wb = b.matchweek === null ? Infinity : b.matchweek;
    if (wa !== wb) return wa - wb;
    if (a.match_date !== b.match_date) return a.match_date < b.match_date ? -1 : 1;
    return String(a.match_id).localeCompare(String(b.match_id), undefined, { numeric: true });
  }

  function normalizeIndex(raw) {
    if (!Array.isArray(raw)) throw new LoadError('the match index is not a list');
    const entries = [];
    raw.forEach(function (e) {
      if (!e || typeof e !== 'object' || e.match_id === undefined || e.match_id === null || typeof e.path !== 'string') return;
      entries.push({
        season: e.season === null || e.season === undefined ? '' : String(e.season),
        match_id: e.match_id,
        match_date: e.match_date === null || e.match_date === undefined ? '' : String(e.match_date),
        matchweek: toNumber(e.matchweek),
        home: String(e.home === null || e.home === undefined ? '' : e.home),
        away: String(e.away === null || e.away === undefined ? '' : e.away),
        home_score: toNumber(e.home_score),
        away_score: toNumber(e.away_score),
        methods: Array.isArray(e.methods) ? e.methods.map(String) : [],
        path: e.path
      });
    });
    if (entries.length === 0) throw new LoadError('the match index lists no matches');
    entries.sort(byWeekThenDate);
    return entries;
  }

  function checkPayload(payload, url) {
    const ok = payload && typeof payload === 'object' && Array.isArray(payload.players) && payload.players.length > 0 &&
      payload.home && typeof payload.home === 'object' && payload.away && typeof payload.away === 'object' &&
      Array.isArray(payload.methods) && payload.methods.length > 0;
    if (!ok) throw new LoadError(url + ' does not look like a match payload');
    payload.players.forEach(function (p) {
      if (!p.ratings || typeof p.ratings !== 'object') p.ratings = {};
      if (p.side !== 'home' && p.side !== 'away') p.side = p.team_id === payload.away.team_id ? 'away' : 'home';
    });
    return payload;
  }

  async function loadPayload(entry) {
    const url = new URL(entry.path, new URL(state.indexUrl, document.baseURI)).href;
    if (state.cache.has(url)) return state.cache.get(url);
    const payload = checkPayload(await fetchJson(url), url);
    state.cache.set(url, payload);
    return payload;
  }

  async function useIndex(url, isSample) {
    const entries = normalizeIndex(await fetchJson(url, { cache: 'no-cache' }));
    state.index = entries;
    state.indexUrl = url;
    state.sample = isSample;
    state.cache.clear();
  }

  function describeFailure(err) {
    return err && err.message ? err.message : String(err);
  }

  /* ================================================================ picker */
  function seasons() {
    const seen = [];
    state.index.forEach(function (e) { if (seen.indexOf(e.season) === -1) seen.push(e.season); });
    return seen;
  }
  function visibleEntries() {
    return state.index.filter(function (e) {
      if (state.filters.season && e.season !== state.filters.season) return false;
      if (state.filters.team && e.home !== state.filters.team && e.away !== state.filters.team) return false;
      return true;
    });
  }
  function findEntry(matchId) {
    for (let i = 0; i < state.index.length; i++) if (String(state.index[i].match_id) === String(matchId)) return state.index[i];
    return null;
  }

  function optionText(e) {
    const score = e.home_score !== null && e.away_score !== null ? e.home_score + '–' + e.away_score : 'v';
    return [fmtDate(e.match_date), e.home + ' ' + score + ' ' + e.away].filter(Boolean).join(' · ');
  }

  function buildFilters() {
    const all = seasons();
    clear(ui.seasonSelect);
    all.forEach(function (season) { ui.seasonSelect.appendChild(h('option', { value: season, text: season || 'Season' })); });
    ui.seasonField.hidden = all.length < 2;
    if (!state.filters.season || all.indexOf(state.filters.season) === -1) state.filters.season = all[all.length - 1] || '';
    ui.seasonSelect.value = state.filters.season;

    const teams = [];
    state.index.forEach(function (e) {
      if (e.season !== state.filters.season) return;
      [e.home, e.away].forEach(function (t) { if (t && teams.indexOf(t) === -1) teams.push(t); });
    });
    teams.sort(function (a, b) { return a.localeCompare(b); });
    if (state.filters.team && teams.indexOf(state.filters.team) === -1) state.filters.team = '';
    clear(ui.teamSelect);
    ui.teamSelect.appendChild(h('option', { value: '', text: 'All teams' }));
    teams.forEach(function (t) { ui.teamSelect.appendChild(h('option', { value: t, text: t })); });
    ui.teamSelect.value = state.filters.team;
  }

  function buildMatchOptions() {
    const list = visibleEntries();
    const select = clear(ui.matchSelect);
    const groups = [];
    list.forEach(function (e) {
      const key = e.season + '|' + e.matchweek;
      let group = groups.length ? groups[groups.length - 1] : null;
      if (!group || group.key !== key) { group = { key: key, matchweek: e.matchweek, entries: [] }; groups.push(group); }
      group.entries.push(e);
    });
    groups.forEach(function (group) {
      const dates = group.entries.map(function (e) { return e.match_date; }).filter(Boolean).sort();
      const range = dates.length ? fmtDateRange(dates[0], dates[dates.length - 1]) : '';
      const title = group.matchweek === null ? 'Other matches' : 'Matchweek ' + group.matchweek;
      select.appendChild(h('optgroup', { label: range ? title + ' · ' + range : title },
        group.entries.map(function (e) { return h('option', { value: String(e.match_id), text: optionText(e) }); })));
    });
    select.disabled = list.length === 0;
    syncPicker();
  }

  function syncPicker() {
    if (state.matchId !== null) ui.matchSelect.value = String(state.matchId);
    const list = visibleEntries();
    let at = -1;
    list.forEach(function (e, i) { if (String(e.match_id) === String(state.matchId)) at = i; });
    ui.prev.disabled = at <= 0;
    ui.next.disabled = at === -1 || at >= list.length - 1;
  }

  function step(delta) {
    const list = visibleEntries();
    let at = -1;
    list.forEach(function (e, i) { if (String(e.match_id) === String(state.matchId)) at = i; });
    const target = list[at + delta];
    if (target) selectMatch(target.match_id);
  }

  /* ================================================================ url hash + storage */
  function readHash() {
    try {
      const params = new URLSearchParams(String(location.hash || '').replace(/^#/, ''));
      return { match: params.get('match'), method: params.get('method'), team: params.get('team') || '' };
    } catch (err) { return { match: null, method: null, team: '' }; }
  }
  function writeHash() {
    try {
      const params = new URLSearchParams();
      if (state.matchId !== null) params.set('match', String(state.matchId));
      if (state.payload && state.payload.methods.length > 1 && state.method) params.set('method', state.method);
      if (state.filters.team) params.set('team', state.filters.team);
      history.replaceState(null, '', '#' + params.toString());
    } catch (err) { /* file:// or sandboxed frames may refuse: not essential */ }
  }
  function storedMethod() {
    try { return window.localStorage.getItem(STORAGE_KEY); } catch (err) { return null; }
  }
  function storeMethod(method) {
    try { window.localStorage.setItem(STORAGE_KEY, method); } catch (err) { /* private mode: ignore */ }
  }

  /* ================================================================ status + errors */
  function setState(name) { document.body.setAttribute('data-state', name); }
  function announce(text) { ui.srStatus.textContent = text; }

  function showBanner() {
    if (!state.sample) { ui.banner.hidden = true; clear(ui.banner); return; }
    clear(ui.banner).appendChild(h('span', null,
      h('strong', { text: 'Showing synthetic sample data. ' }),
      'These players, teams and ratings are made up to demonstrate the page; they are not real.',
      h('span', { class: 'why', text: 'Real ratings are not available yet (' + (state.fallbackReason || 'data/index.json was not found') + '). Create them with python -m src.pipeline.export_web.' })));
    ui.banner.hidden = false;
  }

  function showNotice(nodes) {
    clear(ui.notice);
    appendAll(ui.notice, nodes);
    ui.notice.hidden = false;
  }
  function hideNotice() { ui.notice.hidden = true; clear(ui.notice); }

  function showFatal(dataError, sampleError) {
    setState('error');
    const onFile = window.location && window.location.protocol === 'file:';
    clear(ui.head).appendChild(h('p', { class: 'loading', text: 'No match data could be loaded.' }));
    ui.matchSelect.disabled = true;
    clear(ui.matchSelect).appendChild(h('option', { text: 'No matches available' }));
    showNotice([
      h('strong', { text: 'Could not load match data. ' }),
      'Tried ' + DATA_INDEX_URL + ' (' + describeFailure(dataError) + ') and ' + SAMPLE_INDEX_URL + ' (' + describeFailure(sampleError) + '). ',
      onFile ? 'Browsers block fetch() on file:// pages, so open this folder through a web server: ' : 'Serve this folder over HTTP: ',
      h('code', { text: 'python -m http.server 8810 --directory web' }),
      '. Generate the real data with ', h('code', { text: 'python -m src.pipeline.export_web' }), '.'
    ]);
  }

  function showMatchError(entry, err) {
    setState('error');
    state.payload = null;
    clear(ui.markers);
    clear(ui.tableBody);
    $$('.sub-list').forEach(clear);
    clear(ui.head).appendChild(h('p', { class: 'loading', text: 'This match could not be loaded.' }));
    showNotice([
      h('strong', { text: 'Could not load ' + optionText(entry) + '. ' }),
      describeFailure(err) + '. ',
      h('button', { type: 'button', class: 'step retry', onclick: function () { selectMatch(entry.match_id); } , text: 'Try again' })
    ]);
  }

  /* ================================================================ actions */
  /* A match picked via the URL, or after a filter change, must not be hidden by the season / team filters. */
  function ensureVisible(entry) {
    let changed = false;
    if (state.filters.season !== entry.season) { state.filters.season = entry.season; changed = true; }
    if (state.filters.team && entry.home !== state.filters.team && entry.away !== state.filters.team) {
      state.filters.team = '';
      changed = true;
    }
    if (changed) { buildFilters(); buildMatchOptions(); }
  }

  async function selectMatch(matchId) {
    const entry = findEntry(matchId);
    if (!entry) return;
    ensureVisible(entry);
    state.matchId = entry.match_id;
    syncPicker();
    const token = ++state.token;
    setState('loading');
    try {
      const payload = await loadPayload(entry);
      if (token !== state.token) return;                   // a newer selection took over
      state.payload = payload;
      if (payload.methods.indexOf(state.method) === -1) state.method = payload.methods[0];
      state.placed = layoutMatch(payload);
      hideNotice();
      renderAll();
      writeHash();
      setState('ready');
      announce(payload.home.team + ' ' + fmtScore(payload.home.score) + ', ' + payload.away.team + ' ' + fmtScore(payload.away.score) + '. Showing ' + methodLabel(state.method) + ' ratings.');
    } catch (err) {
      if (token !== state.token) return;
      showMatchError(entry, err);
    }
  }

  function setMethod(method) {
    if (!state.payload || state.payload.methods.indexOf(method) === -1 || method === state.method) return;
    state.method = method;
    storeMethod(method);
    renderAll();
    writeHash();
    announce('Showing ' + methodLabel(method) + ' ratings. The player table is re-sorted.');
  }

  function onFilterChange() {
    buildFilters();
    buildMatchOptions();
    const list = visibleEntries();
    const stillThere = list.some(function (e) { return String(e.match_id) === String(state.matchId); });
    if (!stillThere && list.length) selectMatch(list[0].match_id);
    else writeHash();
  }

  function wireEvents() {
    ui.matchSelect.addEventListener('change', function () { selectMatch(ui.matchSelect.value); });
    ui.teamSelect.addEventListener('change', function () { state.filters.team = ui.teamSelect.value; onFilterChange(); });
    ui.seasonSelect.addEventListener('change', function () { state.filters.season = ui.seasonSelect.value; state.filters.team = ''; onFilterChange(); });
    ui.prev.addEventListener('click', function () { step(-1); });
    ui.next.addEventListener('click', function () { step(1); });
    ui.toggle.addEventListener('change', function (event) {
      if (event.target && event.target.name === 'method') setMethod(event.target.value);
    });
    document.addEventListener('mouseover', onPointerOver);
    document.addEventListener('focusin', onPointerOver);
    document.addEventListener('mouseout', function (event) { if (!event.relatedTarget) setHighlight(null); });
    document.addEventListener('focusout', function () { setHighlight(null); });
    window.addEventListener('hashchange', function () {
      const wanted = readHash();
      if (wanted.method && state.payload && state.payload.methods.indexOf(wanted.method) !== -1) setMethod(wanted.method);
      if (wanted.match && String(wanted.match) !== String(state.matchId) && findEntry(wanted.match)) selectMatch(wanted.match);
    });
  }

  function cacheDom() {
    ui.head = $('#match-head');
    ui.banner = $('#banner');
    ui.notice = $('#notice');
    ui.seasonField = $('#field-season');
    ui.seasonSelect = $('#season-select');
    ui.teamSelect = $('#team-select');
    ui.matchSelect = $('#match-select');
    ui.prev = $('#prev-match');
    ui.next = $('#next-match');
    ui.toggle = $('#method-toggle');
    ui.markers = $('#markers');
    ui.noteAway = $('#note-away');
    ui.noteHome = $('#note-home');
    ui.noteAwayForm = $('#note-away-form');
    ui.noteHomeForm = $('#note-home-form');
    ui.subsHome = $('#subs-home');
    ui.subsAway = $('#subs-away');
    ui.tableBody = $('#players-table tbody');
    ui.tableCaption = $('#table-caption');
    ui.ratingHeader = $('#th-rating');
    ui.sortNote = $('#sort-note');
    ui.methodNote = $('#method-note');
    ui.gkNote = $('#gk-note');
    ui.lineupsTitle = $('#lineups-title');
    ui.credit = $('#credit');
    ui.creditDefault = Array.prototype.slice.call(ui.credit.childNodes).map(function (node) { return node.cloneNode(true); });
    ui.srStatus = $('#sr-status');
  }

  async function init() {
    cacheDom();
    wireEvents();
    const wanted = readHash();
    state.method = wanted.method || storedMethod() || 'baseline';
    state.filters.team = wanted.team;
    state.fallbackReason = '';

    try {
      await useIndex(DATA_INDEX_URL, false);
    } catch (dataError) {
      state.fallbackReason = describeFailure(dataError);
      try {
        await useIndex(SAMPLE_INDEX_URL, true);
      } catch (sampleError) {
        showFatal(dataError, sampleError);
        return;
      }
    }

    showBanner();
    const target = wanted.match !== null ? findEntry(wanted.match) : null;
    if (target) state.filters.season = target.season;
    buildFilters();
    buildMatchOptions();
    await selectMatch(target ? target.match_id : (visibleEntries()[0] || state.index[0]).match_id);
  }

  /* Exposed for tests and debugging only. */
  window.MatchRatings = {
    ratingBand: ratingBand, shortName: shortName, layoutTeam: layoutTeam, resolveOverlaps: resolveOverlaps,
    fmtFormation: fmtFormation, fmtDate: fmtDate, POSITION_LAYOUT: POSITION_LAYOUT, state: state
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
