from langchain_core.messages import HumanMessage
import json

def validate_question(candidate_question, previous_questions, llm_json):
    validator_prompt = f"""
You are a strict interview question validator.

Your job is to determine whether the proposed question is acceptable.

Check the following:
1. The question must contain ONLY ONE question.
2. It must not contain multiple question marks.
3. It must not repeat previously asked questions.
4. It must not explain the topic.
5. It must not contain commentary or instructions.

Previously asked questions:
{json.dumps(previous_questions)}

Candidate question:
{candidate_question}

Return JSON:
{{
  "valid": true or false,
  "reason": "short explanation"
}}
"""
    val_response = llm_json.invoke([HumanMessage(content=validator_prompt)]).content
    val_data = json.loads(val_response)
    print("----------------------- Validator -----------------------")
    print(val_data)
    # can throw json error as well capture in main
    return val_data