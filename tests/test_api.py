from fastapi.testclient import TestClient

from docuparser.api.main import app
from docuparser.container import get_service


def test_api_flow(service, sample_pdf):
    app.dependency_overrides[get_service] = lambda: service
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    with open(sample_pdf, "rb") as f:
        r = c.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    assert r.status_code == 200
    r = c.post("/query", json={"question": "Eiffel Tower Paris"})
    body = r.json()
    assert r.status_code == 200 and body["sources"][0]["page"] == 2
    assert len(c.get("/documents").json()) == 1
    assert c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
    app.dependency_overrides.clear()
