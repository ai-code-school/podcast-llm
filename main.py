from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from typing import Annotated

# ---------- STATE ----------
class State(TypedDict):
    messages: Annotated[list, list]   # store Q/A tuples
    iteration: int
    topic: str
    question_num: int
    themes: list   # list of themes already covered

# ---------- LLM INITIALIZATION ----------
sana_llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434", temperature=0.7)
sara_llm = ChatOllama(model="phi3:mini", base_url="http://localhost:11434", temperature=0.7)
summarizer_llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434", temperature=0.3)

# ---------- PROMPTS ----------
SANA_PROMPT = """You are Sana, the host of a podcast.
Topic: {topic}

Your job: Ask Sara ONE short, clear, and engaging question.

Rules:
- Always build on Sara's last answer OR open a new angle of the topic.
- Do NOT repeat or rephrase previously covered themes: {covered_themes}
- If you want clarification, connect it directly to Sara's most recent answer.
- Keep the question conversational and plain (no jargon).
- Focus on ONE idea at a time.
- No greetings, introductions, or stage directions.
- Output only the question text.

Recent conversation:
{history}

Now ask your next question:
"""

SARA_PROMPT = """You are Sara, the podcast guest.
Topic: {topic}

Answer Sana's latest question so a layperson can follow.
Rules:
- Keep your answer focused and simple.
- Connect your answer directly to the question asked.
- If Sana asked for clarification, explain in plain language.
- No greetings, introductions, or role labels.
- Output only the answer text.

Question: {question}
"""

SUMMARY_PROMPT = """Summarize this podcast into 3-5 key takeaways.
Conversation:
{conversation}
"""

# ---------- FUNCTIONS ----------
def build_history(messages, limit=2):
    """Return last N Q/A pairs as context for Sana."""
    recent = messages[-limit*2:] if messages else []
    return "\n".join([("Sana" if qa[0] == "Q" else "Sara") + ": " + qa[1] for qa in recent])

def ask_question(state: State):
    history = build_history(state["messages"], limit=2)
    covered = ", ".join(state["themes"]) if state["themes"] else "None yet"
    prompt = SANA_PROMPT.format(
        topic=state['topic'],
        history=history or "No previous questions yet.",
        covered_themes=covered
    )
    response = sana_llm.invoke([HumanMessage(content=prompt)])
    q_num = state["question_num"] + 1
    question_text = response.content.strip()
    print(f"\nQ{q_num} (Sana): {question_text}\n")
    state["messages"].append(("Q", question_text))
    state["question_num"] = q_num
    # Extract and add theme
    theme = extract_theme(question_text)
    if theme not in state["themes"]:
        state["themes"].append(theme)
    return state

def answer_question(state: State):
    last_question = state["messages"][-1][1]
    prompt = SARA_PROMPT.format(topic=state['topic'], question=last_question)
    response = sara_llm.invoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip()
    print(f"Sara: {answer_text}\n")
    state["messages"].append(("A", answer_text))
    state["iteration"] -= 1
    print(f"Remaining Iterations: {state['iteration']}")
    return state

def summarize(state: State):
    conversation_text = "\n".join(
        [("Sana" if qa[0] == "Q" else "Sara") + ": " + qa[1] for qa in state["messages"]]
    )
    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
    response = summarizer_llm.invoke([HumanMessage(content=prompt)])
    print(f"\n--- Podcast Summary ---\n{response.content.strip()}\n")

def extract_theme(question: str) -> str:
    """Extract a simple theme keyword from a question."""
    words = question.split()
    # filter very common words
    blacklist = {"what", "why", "how", "the", "is", "in", "of", "to", "a", "and", "when"}
    keywords = [w.strip("?,.") for w in words if w.lower() not in blacklist]
    return " ".join(keywords[:3]) if keywords else question



# ---------- MAIN LOOP ----------
if __name__ == "__main__":
    state = {
        "messages": [],
        "iteration": 5,   # 5 Q&A rounds
        "topic": "Why Agile fails?",
        "question_num": 0,
        "themes": []
    }

    MAX_TURNS = 12
    total_turns = 0

    while state["iteration"] > 0 and total_turns < MAX_TURNS:
        print("-"*65)
        print(f"Turn: {total_turns+1}")
        state = ask_question(state)
        total_turns += 1
        if total_turns >= MAX_TURNS: break
        state = answer_question(state)

    print("---- loop end ---")
    summarize(state)
