from docuparser.ingestion.pipeline import file_hash


def test_ingest_and_retrieve_text(service, sample_pdf):
    res = service.ingest(sample_pdf)
    assert res.text_chunks >= 2
    ans = service.ask("Where is the Eiffel Tower?")
    top = ans.sources[0]
    assert "Eiffel" in top.chunk.text
    assert top.chunk.metadata["page"] == 2
    assert top.chunk.metadata["source"] == "sample.pdf"


def test_image_is_captioned_and_searchable(service, sample_pdf):
    res = service.ingest(sample_pdf)
    assert res.image_chunks == 1
    ans = service.ask("red square picture", top_k=1)
    assert ans.sources[0].chunk.metadata["chunk_type"] == "image"


def test_reingest_is_idempotent(service, sample_pdf):
    service.ingest(sample_pdf)
    n1 = service.list_documents()[0]["chunks"]
    service.ingest(sample_pdf)
    docs = service.list_documents()
    assert len(docs) == 1 and docs[0]["chunks"] == n1


def test_delete(service, sample_pdf):
    service.ingest(sample_pdf)
    service.delete_document(file_hash(sample_pdf))
    assert service.list_documents() == []
    assert service.ask("anything").sources == []


def test_image_mode_none_skips_images(settings, sample_pdf):
    from docuparser.container import build_service
    settings.image_mode = "none"
    assert build_service(settings).ingest(sample_pdf).image_chunks == 0
