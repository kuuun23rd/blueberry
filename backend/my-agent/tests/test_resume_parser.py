import pytest

from resume_parser import ResumeParseError, extract_resume_text, parse_question_bank


def _build_minimal_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF with `text` drawn on it (or a blank page if empty)."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 200 200] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream_content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode() if text else b""
    objects.append(
        b"<< /Length %d >>\nstream\n%s\nendstream"
        % (len(stream_content), stream_content)
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)

    xref_offset = len(out)
    n = len(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % n
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        n,
        xref_offset,
    )
    return bytes(out)


def test_extract_resume_text_returns_page_text() -> None:
    pdf_bytes = _build_minimal_pdf("Hello Resume")
    assert "Hello Resume" in extract_resume_text(pdf_bytes)


def test_extract_resume_text_raises_on_garbage_bytes() -> None:
    with pytest.raises(ResumeParseError):
        extract_resume_text(b"this is not a pdf")


def test_extract_resume_text_raises_when_no_text_layer() -> None:
    pdf_bytes = _build_minimal_pdf("")
    with pytest.raises(ResumeParseError):
        extract_resume_text(pdf_bytes)


def test_parse_question_bank_parses_json_array() -> None:
    questions = parse_question_bank('["Question one?", "Question two?"]')
    assert questions == ["Question one?", "Question two?"]


def test_parse_question_bank_strips_code_fence() -> None:
    questions = parse_question_bank('```json\n["Question one?"]\n```')
    assert questions == ["Question one?"]


def test_parse_question_bank_raises_on_non_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_question_bank("not json at all")


def test_parse_question_bank_raises_on_non_list_json() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_question_bank('{"question": "not a list"}')


def test_parse_question_bank_raises_on_blank_entries() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_question_bank('["Real question?", "   "]')
