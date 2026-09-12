"""Interview question flow, split into two independently testable stages.

Asking a question and deciding what to do with the answer used to live inside
one long system prompt on a single agent. This module splits them:

- ``AskQuestionTask`` is a scoped LiveKit ``AgentTask`` whose only job is to
  ask one question and recognize when the candidate is done answering.
- ``evaluate_answer`` is a plain (non-voice) function that decides whether to
  follow up on the same topic or move to the next question. It makes its own
  direct LLM call rather than talking to the candidate, so it doesn't need a
  full agent/voice turn and can be tested without spinning up a session.

NOTE: ``GENERIC_QUESTIONS`` is a placeholder question bank for early testing.
It is not grounded in the candidate's CV -- that comes from the resume
parsing + question generation work.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from livekit.agents import (
    NOT_GIVEN,
    AgentTask,
    NotGivenOr,
    RunContext,
    function_tool,
    llm,
)

GENERIC_QUESTIONS: list[str] = [
    "Tell me about a time you faced a challenging technical problem and how you approached it.",
    "What tools or technologies are you most comfortable working with for this kind of role?",
    "Describe a situation where you disagreed with a teammate. How did you handle it?",
    "Walk me through how you'd approach designing or debugging a system relevant to this role.",
    "Tell me about a project you're proud of. What was your specific contribution?",
    "How do you prioritize tasks when you have multiple deadlines at once?",
]


@dataclass
class TranscriptEntry:
    """One asked question + the candidate's answer, plus feedback filled in afterward.

    `feedback` starts as None and is populated in bulk, after the whole interview finishes, by
    `generate_transcript_feedback` -- generating feedback per-question as we go would add an
    extra non-voice LLM call (and a little latency) after every single answer, so we do it once
    at the end instead.
    """

    question: str
    answer_summary: str
    feedback: str | None = None


@dataclass
class InterviewState:
    target_role: str | None = None
    remaining_questions: list[str] = field(default_factory=list)
    questions_asked: int = 0
    max_questions: int = 8
    # Populated by the resume byte-stream handler in agent.py, if the candidate uploaded one
    # (see resume_parser.py). Read opportunistically when the question bank is chosen -- there's
    # no synchronization with the upload, so a resume that arrives late is simply missed in favor
    # of GENERIC_QUESTIONS rather than adding latency to wait for it.
    resume_text: str | None = None
    # Every question asked and how the candidate answered it, in order, across the whole
    # interview (including follow-ups). Sent to the frontend at the end (see agent.py) so the
    # candidate can review their full Q&A and per-question feedback on the Summary screen.
    transcript: list[TranscriptEntry] = field(default_factory=list)
    # Every line the agent actually spoke (assistant-role conversation items), in order, for
    # the whole session -- populated by a "conversation_item_added" listener registered in
    # agent.py's my_agent(). AskQuestionTask is instructed to ask each question "in your own
    # natural phrasing" rather than quoting it verbatim, so this is how we record what the
    # interviewer literally said, as opposed to the internal canonical question text it was
    # given. Used by _ask_and_record (agent.py) to fill in each TranscriptEntry.question.
    assistant_lines: list[str] = field(default_factory=list)


@dataclass
class QuestionAnswer:
    question: str
    answer_summary: str


class AskQuestionTask(AgentTask[QuestionAnswer]):
    """Asks the candidate exactly one interview question and captures their answer.

    Scoped narrowly on purpose: it only asks the question and recognizes when
    the candidate has answered. It does not judge the answer's quality or
    decide what happens next -- see ``evaluate_answer`` for that.
    """

    def __init__(
        self, *, question: str, chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN
    ) -> None:
        self._question = question
        super().__init__(
            instructions=textwrap.dedent(
                f"""\
                You are mid-interview, asking the candidate this one specific question:
                "{question}"

                Ask it in your own natural phrasing -- you don't need to quote it verbatim.
                If the candidate asks you to clarify, clarify briefly and keep waiting; do not
                call the tool until they've actually answered.

                Once they've given their answer, call `answer_captured` with a concise summary
                of what they said. Do not evaluate the answer's quality or decide what to ask
                next -- that happens elsewhere. Do not ask a second question yourself.
                """
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=f'Ask the candidate: "{self._question}"'
        )

    @function_tool
    async def answer_captured(self, context: RunContext, summary: str) -> None:
        """Call this once the candidate has finished answering the question.

        Args:
            summary: A concise summary of what the candidate said.
        """
        self.complete(QuestionAnswer(question=self._question, answer_summary=summary))


@dataclass
class FollowUpDecision:
    action: Literal["follow_up", "move_on"]
    next_question: str | None = None


_DECISION_INSTRUCTIONS = """\
You are an interview coach deciding what happens next in a mock interview, right after the \
candidate answered one question. You are not talking to the candidate -- your output is read \
by another system, not a person, so don't phrase it as speech.

Question asked: {question}
Candidate's answer: {answer}

Decide whether to:
- Ask a natural follow-up question that digs deeper into what they just said, because the \
answer was short, vague, or left something interesting unexplored, or
- Move on, because the answer was already detailed and complete.

Respond with EXACTLY one of these two formats and nothing else:
FOLLOW_UP: <the follow-up question to ask, phrased naturally>
MOVE_ON
"""


def parse_followup_decision(text: str) -> FollowUpDecision:
    """Parse the evaluator LLM's constrained-format response.

    Falls back to MOVE_ON for anything that doesn't match the expected format,
    so a malformed model response can never wedge the interview loop.
    """
    match = re.match(r"\s*FOLLOW_UP:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match and match.group(1).strip():
        return FollowUpDecision(
            action="follow_up", next_question=match.group(1).strip()
        )
    return FollowUpDecision(action="move_on")


async def evaluate_answer(
    llm_v: llm.LLM, *, question: str, answer: str
) -> FollowUpDecision:
    """Decide whether to dig deeper on ``question`` or move on, given the candidate's answer.

    This is a plain LLM call, not a voice turn -- it never speaks to the candidate, which keeps
    it fast and lets it be reasoned about (and tested) independently of ``AskQuestionTask``.
    """
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(
        role="system",
        content=_DECISION_INSTRUCTIONS.format(question=question, answer=answer),
    )
    response = await llm_v.chat(chat_ctx=chat_ctx).collect()
    return parse_followup_decision(response.text)


def _strip_code_fence(text: str) -> str:
    """Strip a ```...``` or ```json...``` wrapper some models add despite instructions not to.

    Mirrors the identically-named helper in resume_parser.py -- kept as a separate copy rather
    than a shared import so the two modules stay independent and independently testable.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()


_FEEDBACK_INSTRUCTIONS = """\
You are an experienced interview coach reviewing a candidate's mock interview transcript. You \
are not talking to the candidate directly -- your output is read by another system, not spoken \
aloud or shown as a transcript of speech.

For each numbered question-and-answer pair below, write one short piece of constructive \
feedback (1 to 2 sentences): what the candidate did well, and one concrete thing they could \
improve, if anything. If an answer was already strong and complete, say so briefly rather than \
inventing a criticism.

Respond with ONLY a JSON array of strings, exactly one entry per numbered pair below, in the \
same order, and nothing else.

Transcript:
{transcript_block}
"""


def _format_transcript(transcript: list[TranscriptEntry]) -> str:
    return "\n\n".join(
        f"{i + 1}. Q: {entry.question}\n   A: {entry.answer_summary}"
        for i, entry in enumerate(transcript)
    )


def parse_feedback_list(text: str, *, expected_count: int) -> list[str]:
    """Parse the feedback generator's JSON array response.

    Raises:
        ValueError: if the response isn't a JSON array of exactly `expected_count` non-empty
            strings.
    """
    try:
        data = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Feedback generator did not return valid JSON: {e}") from e

    if (
        not isinstance(data, list)
        or len(data) != expected_count
        or not all(isinstance(f, str) and f.strip() for f in data)
    ):
        raise ValueError(
            "Feedback generator did not return the expected list of feedback strings"
        )

    return [f.strip() for f in data]


async def generate_transcript_feedback(
    llm_v: llm.LLM, transcript: list[TranscriptEntry]
) -> list[str]:
    """Generate one short feedback note per transcript entry, in the same order.

    This is a single plain (non-voice) LLM call over the whole transcript, made once after the
    interview wraps up -- not per-question -- to avoid adding an extra round-trip after every
    single answer.

    Raises:
        ValueError: if the response can't be parsed (see `parse_feedback_list`) -- callers
            should treat this as "no feedback available" rather than blocking on it.
    """
    if not transcript:
        return []
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(
        role="system",
        content=_FEEDBACK_INSTRUCTIONS.format(
            transcript_block=_format_transcript(transcript)
        ),
    )
    response = await llm_v.chat(chat_ctx=chat_ctx).collect()
    return parse_feedback_list(response.text, expected_count=len(transcript))
