import json
from langchain_core.messages import HumanMessage


def generate_question(topic_data, past_topics, llm):
    generator_prompt = f"""
You are a sharp podcast host.

Your job is to generate ONE thought-provoking question for the guest.

Context
Current discussion topic:
{json.dumps(topic_data, indent=2)}

Topics already discussed:
{json.dumps(past_topics)}

Strict Rules:
- Ask EXACTLY ONE question
- The output must contain ONLY ONE sentence
- The sentence must end with ONE question mark
- Do NOT include follow-up questions
- Do NOT chain questions using "and", "also", "or", "what about"
- Do NOT explain the topic
- Do NOT summarize the guest answer
- Do NOT mention formatting rules
- Avoid repeating past topics
- Focus on deeper reasoning, tradeoffs, assumptions, failures, or consequences

Good podcast questions challenge the guest's thinking.

Examples:
"If microservices promise faster delivery, why do so many teams slow down after adopting them?"

"What hidden assumption about scaling usually breaks first when a system grows beyond its original architecture?"

Output:
Return ONLY the question text.
"""

    try:
        candidate_question = llm.invoke([HumanMessage(content=generator_prompt)]).content.strip()
        print("----------------------- Question Generator -----------------------")
        print(candidate_question)
    except Exception as e:
        # Fallback if JSON parsing fails
        candidate_question = "Unknown"
        print("----------------------- Fallback question generator ----------------------- ")
    
    return candidate_question