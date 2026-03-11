import json
from langchain_core.messages import HumanMessage

def extract_topic(guest_answer, llm_json):
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
    
    return topic_data