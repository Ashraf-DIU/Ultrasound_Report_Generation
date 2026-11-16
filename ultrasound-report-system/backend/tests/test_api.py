"""End-to-end API test.

Needs a bundle. Build a fake one first (see backend/README.md), then:
    BUNDLE_DIR=/kaggle/working/web_bundle TEST_IMAGE=/kaggle/test_sheet.jpg pytest -q
"""
import os
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not os.environ.get("BUNDLE_DIR"), reason="BUNDLE_DIR not set")


@pytest.fixture(scope="module")
def client():
    from app.main import STATE, app
    with TestClient(app) as c:
        for _ in range(120):
            if STATE["status"] != "loading":
                break
            time.sleep(0.5)
        assert STATE["status"] == "ready", STATE
        yield c


def _sheet_bytes():
    p = os.environ.get("TEST_IMAGE")
    if p and os.path.exists(p):
        return open(p, "rb").read()
    img = np.zeros((400, 600, 3), np.uint8)
    cv2.circle(img, (300, 200), 120, (120, 120, 120), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    assert client.get("/health").json()["status"] == "ready"


def test_info(client):
    j = client.get("/api/info").json()
    assert j["status"] == "ready" and j["corpus_size"] > 0 and j["k"] == 5


def test_report(client):
    r = client.post("/api/report", files={"file": ("sheet.jpg", _sheet_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["report"]["impression"]
    assert len(j["similar_cases"]) == 5
    assert j["quality"]["n_frames"] >= 1
    assert j["frames"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    assert "NOT a diagnosis" in j["disclaimer"]


def test_rejects_non_image(client):
    r = client.post("/api/report", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_garbage_bytes(client):
    r = client.post("/api/report", files={"file": ("x.jpg", b"notanimage", "image/jpeg")})
    assert r.status_code == 400
