import json
from langchain_core.messages import HumanMessage

def generate_question(topic_data, past_topics, llm):
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
    try:
        candidate_question = llm.invoke([HumanMessage(content=generator_prompt)]).content.strip()
        print("----------------------- Question Generator -----------------------")
        print(candidate_question)
    except Exception as e:
        # Fallback if JSON parsing fails
        candidate_question = "Unknown"
        print("----------------------- Fallback question generator ----------------------- ")
    
    return candidate_question