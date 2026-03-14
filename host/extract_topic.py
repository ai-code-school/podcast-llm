import json
from langchain_core.messages import HumanMessage

def extract_topic(guest_answer, podcast_topic, llm_json):
    extractor_prompt = f"""
You are an expert conversation analyzer for podcasts.

Podcast main topic:
{podcast_topic}

Your job is to extract the **specific discussion point** from the guest answer
within the context of the podcast topic.

Rules:
- Anchor the topic to the podcast main topic
- Be specific, not generic
- Avoid vague words like: technology, innovation, strategy, future, impact
- Focus on the actual claim, idea, argument, or insight being discussed
- Ignore storytelling, filler, or examples
- Extract only meaningful discussion directions

Bad topics:
- "Artificial Intelligence"
- "Startups"
- "Technology trends"

Good topics:
- "Why most AI startups fail due to lack of proprietary data"
- "Tradeoffs between open-source and closed AI models"
- "Why large datasets create defensibility in AI companies"

Return JSON only.

Format:
{{
  "topic": "{podcast_topic}",
  "subtopics": ["narrow aspect 1", "narrow aspect 2", "narrow aspect 3"],
  "keywords": ["important term 1", "important term 2", "important term 3"],
  "possible_followup_angles": [
    "hidden assumptions",
    "tradeoffs or downsides",
    "real world failures",
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

    except Exception:
        topic_data = {
            "topic": podcast_topic,
            "subtopics": [],
            "keywords": [],
            "possible_followup_angles": []
        }
        print("----------------------- Fallback extraction ----------------------- ")

    return topic_data
