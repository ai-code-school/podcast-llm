import json
import re
from typing import List, Tuple, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from host.extract_topic import extract_topic
from host.generate_question import generate_question
from host.validate_question import validate_question
from host.host_question import host_question

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
    topic_data = extract_topic(guest_answer, llm_json)

    # ---------------------------------------------------------
    # STEP 2 & 3: Question Generator & Validator Loop
    # ---------------------------------------------------------
    MAX_RETRIES = 3
    valid_question = ""
    
    for attempt in range(MAX_RETRIES):
        candidate_question = generate_question(topic_data, past_topics, llm)

        try:
            val_data = validate_question(candidate_question, previous_questions, llm_json)

            # Additional hard check on question marks
            if candidate_question.count('?') > 1:
                val_data['valid'] = False
            
            if val_data.get('valid', False):
                valid_question = candidate_question
                break
        except Exception as e:
            print(f"Validation failed: {e}")
            pass # Try again if validation JSON fails

        if not valid_question:
            # Fallback if all retries fail
            valid_question = candidate_question

    # ---------------------------------------------------------
    # STEP 4: Host Output
    # ---------------------------------------------------------
    final_output = host_question(llm, valid_question)
    return final_output, topic_data