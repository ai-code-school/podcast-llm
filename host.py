import json
import re
from typing import List, Tuple, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def run_host_pipeline(
    llm: ChatOllama, 
    guest_answer: str, 
    past_topics: List[str], 
    previous_questions: List[str]
) -> Tuple[str, Dict]:
    """
    Runs the 4-step pipeline to generate a host response.
    Returns a tuple of (host_spoken_question, current_topic_json)
    """

    print("----------------------- Guest Answer -----------------------")
    print(guest_answer)
    # Force JSON format for extraction and validation steps
    llm_json = ChatOllama(
        model=llm.model, 
        base_url=llm.base_url, 
        temperature=llm.temperature,
        format="json"
    )

    # ---------------------------------------------------------
    # STEP 1: Topic Extractor
    # ---------------------------------------------------------
    extractor_prompt = f"""
You are a conversation analyzer.

Your job is to extract the core discussion topic from the guest's answer.

Rules:
- Focus only on the main ideas
- Ignore storytelling, examples, or filler
- Extract the core technical or conceptual discussion
- Do not generate questions

Return JSON only.

Format:
{{
  "topic": "main topic being discussed",
  "subtopics": ["subtopic1", "subtopic2", "subtopic3"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "possible_followup_angles": [
    "assumptions behind the idea",
    "tradeoffs or risks",
    "real-world failures",
    "scaling implications"
  ]
}}

Guest answer:
{guest_answer}
"""
    try:
        topic_response = llm_json.invoke([HumanMessage(content=extractor_prompt)]).content
        print("----------------------- Extraction -----------------------")
        print(topic_response)
        topic_data = json.loads(topic_response)
    except Exception as e:
        # Fallback if JSON parsing fails
        topic_data = {"topic": "Unknown", "subtopics": [], "keywords": [], "possible_followup_angles": []}
        print("----------------------- Fallback extraction ----------------------- ")
        


    # ---------------------------------------------------------
    # STEP 2 & 3: Question Generator & Validator Loop
    # ---------------------------------------------------------
    MAX_RETRIES = 3
    valid_question = ""
    
    for attempt in range(MAX_RETRIES):
        generator_prompt = f"""
You are a podcast host question generator.

Your job is to create ONE intelligent question for a podcast.

Context:
Current topic:
{json.dumps(topic_data, indent=2)}

Topics already discussed:
{json.dumps(past_topics)}

Rules:
- Ask exactly ONE question
- Never ask multiple questions
- Never explain the topic
- Do not summarize the guest answer
- Do not give advice
- Do not mention formatting rules
- Avoid repeating previously discussed topics
- Focus on deeper reasoning, tradeoffs, assumptions, or consequences

Good questions challenge thinking.

Examples of good questions:
"If microservices promise faster delivery, why do so many organizations slow down after adopting them?"

"What assumption about scaling breaks first when a system grows beyond its original architecture?"

Output:
Return only one question.
"""
        candidate_question = llm.invoke([HumanMessage(content=generator_prompt)]).content.strip()
        print("----------------------- Question Generator -----------------------")
        print(candidate_question)
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
        try:
            val_response = llm_json.invoke([HumanMessage(content=validator_prompt)]).content
            val_data = json.loads(val_response)
            print("----------------------- Validator -----------------------")
            print(val_data)
            
            # Additional hard check on question marks
            if candidate_question.count('?') > 1:
                val_data['valid'] = False
                
            if val_data.get('valid', False):
                valid_question = candidate_question
                break
        except Exception:
            pass # Try again if validation JSON fails

    if not valid_question:
        # Fallback if all retries fail
        valid_question = candidate_question

    # ---------------------------------------------------------
    # STEP 4: Host Output
    # ---------------------------------------------------------
    host_prompt = f"""
You are a technical podcast host.

Speak the following question naturally.

Rules:
- Ask the question exactly as written
- Do not add commentary
- Do not add explanations
- Do not prefix with any name

Question:
{valid_question}
"""
    # We yield from this step so it can be streamed in app.py if desired, 
    # but the current architecture streams from the main loop. 
    # Since app.py streams characters from the final string, we just return the final string here.
    final_output = llm.invoke([HumanMessage(content=host_prompt)]).content.strip()
    print("----------------------- Host Output -----------------------")
    print(final_output) 
    # Clean up any weird prefixes just in case
    final_output = re.sub(r'^(Host|Stella|Question|Q):\s*', '', final_output, flags=re.IGNORECASE)
    
    return final_output, topic_data
