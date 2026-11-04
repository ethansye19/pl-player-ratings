"""HttpFplClient without the network: element-summary requests, mirroring, request spacing, rate
limits and the User-Agent. The HTTP layer is replaced by SYNTHETIC responses."""
import json

import pytest
import requests

from src.pipeline import fpl
from src.pipeline.fpl import BASE_URL, HttpFplClient


class _Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Clock:
    """Stands in for the time module: monotonic() only moves when the client sleeps or work is simulated."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(fpl, "time", c)
    return c


def _serve(monkeypatch, client, responses):
    calls = []

    def get(url, timeout):
        calls.append(url)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(client.session, "get", get)
    return calls


SUMMARY = {"history": [{"element": 305, "fixture": 71, "round": 8, "opponent_team": 9, "was_home": True, "minutes": 90, "starts": 1}],
           "fixtures": [], "history_past": []}


def test_element_summary_requests_the_player_and_mirrors_the_response(tmp_path, monkeypatch, clock):
    client = HttpFplClient(cache_dir=tmp_path)
    calls = _serve(monkeypatch, client, [_Response(payload=SUMMARY)])
    assert client.element_summary(305) == SUMMARY
    assert calls == [BASE_URL + "element-summary/305/"]
    assert json.loads((tmp_path / "element-summary" / "305.json").read_text(encoding="utf-8")) == SUMMARY


def test_requests_are_spaced_out(monkeypatch, clock):
    client = HttpFplClient(min_interval=1.0)
    _serve(monkeypatch, client, [_Response(payload=SUMMARY), _Response(payload=SUMMARY), _Response(payload=SUMMARY)])
    client.element_summary(1)
    clock.now += 0.25  # some work between the requests
    client.element_summary(2)
    client.element_summary(3)
    assert clock.sleeps == [0.75, 1.0]
    assert client.requests_made == 3


def test_a_rate_limit_is_waited_out_as_the_server_asks(monkeypatch, clock):
    client = HttpFplClient(min_interval=0)
    calls = _serve(monkeypatch, client, [_Response(429, headers={"Retry-After": "7"}), _Response(payload=SUMMARY)])
    assert client.element_summary(305) == SUMMARY
    assert len(calls) == 2 and 7.0 in clock.sleeps


def test_a_request_that_keeps_failing_raises_after_three_attempts(monkeypatch, clock):
    client = HttpFplClient(min_interval=0)
    calls = _serve(monkeypatch, client, [requests.ConnectionError("down")] * 3)
    with pytest.raises(RuntimeError, match="element-summary/305/"):
        client.element_summary(305)
    assert len(calls) == 3 and clock.sleeps == [2, 4]


def test_the_user_agent_is_generic():
    agent = HttpFplClient().session.headers["User-Agent"]
    assert agent == fpl.USER_AGENT
    assert "@" not in agent and "contact" not in agent.lower() and "http" not in agent.lower()
