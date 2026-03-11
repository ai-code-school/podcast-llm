from langchain_core.messages import HumanMessage
import re

def host_question(llm, topic_data):
    host_prompt = f"""
    You are a technical podcast host.

    Speak the following question naturally.

    Rules:
    - Ask the question to the point and only question
    - Do not add commentary
    - Do not add explanations
    - Do not prefix with any name

    Question:
    {topic_data}
    """
    # We yield from this step so it can be streamed in app.py if desired, 
    # but the current architecture streams from the main loop. 
    # Since app.py streams characters from the final string, we just return the final string here.
    final_output = llm.invoke([HumanMessage(content=host_prompt)]).content.strip()
    print("----------------------- Host Output -----------------------")
    print(final_output) 
    # Clean up any weird prefixes just in case
    final_output = re.sub(r'^(Host|Stella|Question|Q):\s*', '', final_output, flags=re.IGNORECASE)
    
    return final_output