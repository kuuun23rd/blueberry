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
