import fitz
import pytest

from docuparser.config import Settings
from docuparser.container import build_service


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, embedding_provider="fake", llm_provider="none",
                    image_mode="caption", caption_provider="fake", _env_file=None)


@pytest.fixture
def service(settings):
    return build_service(settings)


@pytest.fixture
def sample_pdf(tmp_path):
    """2-page PDF: text on both pages, a 200x200 image on page 2."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (220, 20, 20))
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Photosynthesis converts sunlight into chemical energy in plants.")
    p2 = doc.new_page()
    p2.insert_text((72, 100), "The Eiffel Tower is located in Paris and was completed in 1889.")
    p2.insert_image(fitz.Rect(72, 150, 272, 350), pixmap=pix)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return path
