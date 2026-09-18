"""Resume parsing and CV-grounded question generation (Week 4).

Two stages, kept separate and independently testable, mirroring the ask/evaluate
split in interview_flow.py:

- `extract_resume_text` turns an uploaded PDF's raw bytes into plain text, using
  pypdf. (pypdf is the actively-maintained successor to PyPDF2 -- PyPDF2 itself
  is deprecated upstream in favor of it, so we use pypdf here instead.)
- `generate_question_bank` takes that text plus the candidate's target role and
  asks the LLM for a grounded question set (behavioral, technical, role-specific,
  project deep-dive), per the PRD. Like `evaluate_answer` in interview_flow.py,
  this is a plain (non-voice) LLM call -- it produces data, not something spoken
  to the candidate.
"""

from __future__ import annotations

import io
import json

from livekit.agents import llm
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ResumeParseError(Exception):
    """Raised when the uploaded file can't be read as a resume."""


def extract_resume_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a resume PDF's raw bytes.

    Raises:
        ResumeParseError: if the bytes aren't a readable PDF, or no extractable
            text is found (e.g. a scanned image with no text layer).
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as e:
        raise ResumeParseError(f"Could not read file as a PDF: {e}") from e

    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ResumeParseError(
            "No extractable text found in the PDF (it may be a scanned image)."
        )
    return text


_QUESTION_GEN_INSTRUCTIONS = """\
You are preparing a mock interview for a candidate targeting this role: {target_role}

Here is the candidate's resume:
---
{resume_text}
---

Generate 6 to 8 interview questions grounded in this resume and role, mixing:
- Behavioral questions about specific experience mentioned in the resume
- Technical questions appropriate to the role and the candidate's apparent seniority
- Role-specific questions
- At least one project deep-dive question about a specific project named in the resume

Respond with ONLY a JSON array of strings, one per question, and nothing else. Do not include
numbering, markdown, or commentary -- just the JSON array.
"""


def _strip_code_fence(text: str) -> str:
    """Strip a ```...``` or ```json...``` wrapper some models add despite instructions not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()


def parse_question_bank(text: str) -> list[str]:
    """Parse the question-generator LLM's JSON array response.

    Raises:
        ValueError: if the response isn't a JSON array of non-empty strings.
    """
    try:
        data = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Question generator did not return valid JSON: {e}") from e

    if not isinstance(data, list) or not all(
        isinstance(q, str) and q.strip() for q in data
    ):
        raise ValueError("Question generator did not return a JSON array of questions")

    return [q.strip() for q in data]


async def generate_question_bank(
    llm_v: llm.LLM, *, resume_text: str, target_role: str
) -> list[str]:
    """Generate a CV-grounded question bank for the interview.

    Raises:
        ValueError: if the LLM's response can't be parsed as a question list
            (see `parse_question_bank`) -- callers should fall back to
            GENERIC_QUESTIONS in that case rather than starting an interview
            with no questions.
    """
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(
        role="system",
        content=_QUESTION_GEN_INSTRUCTIONS.format(
            target_role=target_role, resume_text=resume_text
        ),
    )
    response = await llm_v.chat(chat_ctx=chat_ctx).collect()
    return parse_question_bank(response.text)


_ROLE_QUESTION_GEN_INSTRUCTIONS = """\
You are preparing a mock interview for a candidate targeting this role: {target_role}

No resume was provided, so generate a general-purpose question set that still fits this \
specific role, mixing:
- Behavioral questions relevant to the role
- Questions about the skills and day-to-day work the role actually involves -- ask about \
technical problem-solving ONLY if the role is a technical one (e.g. software engineering, \
data, IT); for non-technical roles (e.g. sales, marketing, HR, operations, design, \
management, finance), ask about the non-technical skills and scenarios that role actually \
requires instead
- Role-specific situational questions

Generate 6 to 8 questions total.

Respond with ONLY a JSON array of strings, one per question, and nothing else. Do not include
numbering, markdown, or commentary -- just the JSON array.
"""


async def generate_role_questions(llm_v: llm.LLM, *, target_role: str) -> list[str]:
    """Generate a role-aware question bank when no resume was uploaded to ground one in.

    Raises:
        ValueError: if the LLM's response can't be parsed as a question list
            (see `parse_question_bank`) -- callers should fall back to
            GENERIC_QUESTIONS in that case rather than starting an interview
            with no questions.
    """
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(
        role="system",
        content=_ROLE_QUESTION_GEN_INSTRUCTIONS.format(target_role=target_role),
    )
    response = await llm_v.chat(chat_ctx=chat_ctx).collect()
    return parse_question_bank(response.text)
