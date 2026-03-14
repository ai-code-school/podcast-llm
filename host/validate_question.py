from langchain_core.messages import HumanMessage
import json


def basic_question_validation(candidate_question, previous_questions):

    q = candidate_question.strip()

    # must end with one ?
    if q.count("?") != 1 or not q.endswith("?"):
        return False, "Must contain exactly one question mark."

    # avoid extremely long questions
    if len(q.split()) > 35:
        return False, "Question too long."

    # simple duplicate check
    if q.lower() in [p.lower() for p in previous_questions]:
        return False, "Duplicate question."

    return True, None


def validate_question(candidate_question, previous_questions, llm_json):

    ok, reason = basic_question_validation(candidate_question, previous_questions)
    if not ok:
        return {"valid": False, "reason": reason}

    validator_prompt = f"""
You are a strict podcast question validator.

Decide if the candidate question is acceptable.

Rules:
- It must be ONE clear question.
- It must not repeat previously asked questions in meaning.
- It must not explain the topic.
- It must not contain commentary.

Previously asked questions:
{json.dumps(previous_questions)}

Candidate question:
{candidate_question}

Return JSON only.

{{
  "valid": true or false,
  "reason": "short explanation"
}}
"""

    response = llm_json.invoke([HumanMessage(content=validator_prompt)]).content
    print("----------------------- Validator -----------------------")
    print(response)

    return json.loads(response)

