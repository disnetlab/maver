"""The HTTP example should report paused runs and failed polls promptly."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "run_incident.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_incident_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, tmp_path, response):
    module = _module()
    alert = tmp_path / "alert.json"
    alert.write_text('{"alertname": "Test"}', encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_arguments",
        lambda: SimpleNamespace(
            alert=str(alert), url="http://test", target_stage="full", timeout=1, poll=0
        ),
    )

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"session_id": "incident-1"})
        return response

    client_type = httpx.Client
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )
    return module.main()


def test_reconciliation_stops_polling(monkeypatch, tmp_path, capsys):
    result = _run(
        monkeypatch,
        tmp_path,
        httpx.Response(
            200, json={"run_status": "needs_reconciliation", "current_stage": "execution"}
        ),
    )
    assert result == 0
    output = capsys.readouterr()
    assert '"run_status": "needs_reconciliation"' in output.out
    assert "operator reconciliation required" in output.err
    assert "still running" not in output.err


def test_poll_http_error_is_raised(monkeypatch, tmp_path):
    with pytest.raises(httpx.HTTPStatusError, match="503 Service Unavailable"):
        _run(monkeypatch, tmp_path, httpx.Response(503, json={"detail": "unavailable"}))
