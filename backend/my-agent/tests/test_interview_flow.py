from interview_flow import parse_followup_decision


def test_parses_follow_up_with_question() -> None:
    decision = parse_followup_decision(
        "FOLLOW_UP: What made that bug so hard to track down?"
    )
    assert decision.action == "follow_up"
    assert decision.next_question == "What made that bug so hard to track down?"


def test_parses_follow_up_case_insensitively_and_strips_whitespace() -> None:
    decision = parse_followup_decision("  follow_up:   Tell me more.  \n")
    assert decision.action == "follow_up"
    assert decision.next_question == "Tell me more."


def test_parses_move_on() -> None:
    decision = parse_followup_decision("MOVE_ON")
    assert decision.action == "move_on"
    assert decision.next_question is None


def test_falls_back_to_move_on_for_empty_follow_up_question() -> None:
    decision = parse_followup_decision("FOLLOW_UP:    ")
    assert decision.action == "move_on"


def test_falls_back_to_move_on_for_unrecognized_text() -> None:
    decision = parse_followup_decision(
        "I think we should probably ask something else next."
    )
    assert decision.action == "move_on"
    assert decision.next_question is None
