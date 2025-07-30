from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated

# ---------- STATE ----------
class State(TypedDict):
    messages: Annotated[list, list]
    iteration: int
    topic: str
    question_num: int

# ---------- LLM INITIALIZATION ----------
sana_llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434", temperature=0.7)
sara_llm = ChatOllama(model="phi3:mini", base_url="http://localhost:11434", temperature=0.7)
summarizer_llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434", temperature=0.3)

# ---------- PROMPTS ----------
SANA_PROMPT = """You are Sana, the host of a podcast.
Topic: {topic}
Your task: Ask ONE tough and clear question for Sara.
Rules:
- Do NOT say your own name.
- Do NOT greet, welcome, or introduce.
- Do NOT add stage directions like (Sana's voice).
- Ask only the next question, in plain text.
"""

SARA_PROMPT = """You are Sara, the podcast guest.
Topic: {topic}
Your task: Answer Sana's latest question clearly, for a layman.
Rules:
- Do NOT say your own name.
- Do NOT greet or reintroduce.
- Do NOT add stage directions like (Sara's voice).
- Write only the plain answer text, nothing else.
"""

SUMMARY_PROMPT = """Summarize this podcast into 3-5 key takeaways.
Conversation:
{conversation}
"""

# ---------- FUNCTIONS ----------
def ask_question(state: State):
    prompt = SANA_PROMPT.format(topic=state['topic'])
    response = sana_llm.invoke([HumanMessage(content=prompt)])
    q_num = state["question_num"] + 1
    print(f"\nQ{q_num} (Sana): {response.content.strip()}\n")
    state["messages"].append(("Q", response.content.strip()))
    state["question_num"] = q_num
    return state

def answer_question(state: State):
    last_question = state["messages"][-1][1]  # Only last Sana question
    prompt = SARA_PROMPT.format(topic=state['topic']) + f"\nQuestion: {last_question}"
    response = sara_llm.invoke([HumanMessage(content=prompt)])
    print(f"Sara: {response.content.strip()}\n")
    state["messages"].append(("A", response.content.strip()))
    state["iteration"] -= 1
    print(f"Remaining Iterations: {state['iteration']}")
    return state

def summarize(state: State):
    conversation_text = "\n".join(
        [("Sana" if qa[0] == "Q" else "Sara") + ": " + qa[1]
         for qa in state["messages"]]
    )
    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
    response = summarizer_llm.invoke([HumanMessage(content=prompt)])
    print(f"\n--- Podcast Summary ---\n{response.content.strip()}\n")

# ---------- MAIN LOOP ----------
if __name__ == "__main__":
    state = {
        "messages": [],
        "iteration": 3,   # 3 Sana-Sara rounds
        "topic": "Why Agile fails?",
        "question_num": 0
    }

    MAX_TURNS = 10
    total_turns = 0

    while state["iteration"] > 0 and total_turns < MAX_TURNS:
        print("-"*65)
        print(f"Turn: {total_turns+1}")
        state = ask_question(state)
        total_turns += 1
        if total_turns >= MAX_TURNS: break
        state = answer_question(state)
        total_turns += 1

    print("---- loop end ---")
    summarize(state)
