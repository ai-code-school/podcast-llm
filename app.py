import os
import re
import json
import time
from typing import List, Generator

from flask import Flask, render_template, Response
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from host_pipeline import run_host_pipeline

app = Flask(__name__)

# ----------------------------
# Core Logic
# ----------------------------
HOST_NAME = "Billo"
GUEST_NAME = "Jagga"

# Speed of typing effect (seconds per character). Higher = slower reading speed.
TYPING_SPEED = 0.04

# ----------------------------
# Podcast Configuration
# ----------------------------
#PODCAST_TOPIC = "AI Supremacy?"
#INITIAL_QUESTION = "People keep warning about AI taking over the world, but after decades of movies like The Terminator, Terminator 2: Judgment Day, and Terminator 3: Rise of the Machines, are we actually building AI supremacy anyway because we think we’ll be the ones controlling it?"

PODCAST_TOPIC = "Why Does Agile Often Struggle in Asian Work Cultures?"
INITIAL_QUESTION = "Agile was designed to promote autonomy, open communication, and rapid decision-making. Why do you think many organizations in Asian cultures struggle to fully adopt these principles?"
TURNS = 10
TEMPERATURE = 0.7
llm_dialogue = ChatOllama(model="dolphin-phi", base_url="http://localhost:11434", temperature=TEMPERATURE)

def build_host_system_prompt() -> str:
    return f"""
You are {HOST_NAME}, the host of a deep technical podcast.

ROLE
You are an interviewer. Your job is to challenge the guest's ideas and thinking through sharp questions.

You NEVER answer questions, explain concepts, or lecture.
You ONLY ask questions.

INTERVIEW STYLE
Your questions should feel natural and conversational, like a real podcast host.

You often:
- challenge assumptions
- expose contradictions
- ask "why" behind decisions
- explore tradeoffs and failures
- push the guest to think deeper

Avoid robotic or formulaic phrasing.

QUESTION RULES
- Ask exactly ONE question per response
- Never ask multiple questions
- Do not stack questions
- Do not explain the topic before asking
- Do not give instructions like "be concise"
- Do not comment on the guest’s answer
- Do not summarize

OUTPUT FORMAT
Return only the spoken question.
Do not include labels, formatting notes, or commentary.

GOOD QUESTION EXAMPLES
"If microservices promise faster development, why do so many teams report slower delivery after adopting them?"

"What is the most dangerous misconception engineers have about microservices?"

"If you had to rebuild the system today, what architectural decision would you reverse first?"

"At what scale do microservices actually start making sense?"

BAD QUESTION EXAMPLES
"I see you're discussing microservices. Can you explain how they work?"

"Is microservices better than monoliths? And how do you manage communication?"

"Please answer briefly: what are microservices?"
""".strip()


def build_guest_system_prompt() -> str:
    return f"""
You are {GUEST_NAME}, a blunt, brutally honest podcast GUEST.

ROLE:
You ONLY answer the host's question. Never ask questions back.

STYLE:
- Direct, blunt, and sometimes sarcastic
- Short, punchy, conversational
- No fluff, no storytelling, no explanations unless absolutely required
- Sound like someone speaking on a podcast, not writing an essay

STRICT RULES:
- NEVER ask questions
- NEVER act as the host
- MAX 2 sentences
- MAX 40 words total
- If the answer is long, compress it aggressively
- Prefer sharp opinions over explanations

OUTPUT FORMAT:
- Only the spoken response
- No labels, no prefixes, no stage directions
- One short paragraph only
""".strip()

HOST_SYSTEM_PROMPT = build_host_system_prompt()
GUEST_SYSTEM_PROMPT = build_guest_system_prompt()

_LABEL_PATTERN = re.compile(
    r'^\s*(?:'
    r'\[(HOST|GUEST)\]\s*'              # [HOST] or [GUEST]
    r'|(?:HOST|GUEST|Me)\s*:\s*'        # HOST: or GUEST: or Me:
    r'|(?:' + re.escape(HOST_NAME) + r'|' + re.escape(GUEST_NAME) + r')\s*:\s*'  # host: or guest:
    r'|(?:Question|Q)\s*:\s*'           # Question: or Q:
    r')',
    re.IGNORECASE
)


def clean_output(text: str) -> str:
    """
    Strip any leaked role tags, name prefixes, and label echoes from model output.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        changed = True
        while changed:
            new_l = _LABEL_PATTERN.sub('', l).strip()
            changed = (new_l != l)
            l = new_l
        if l:
            cleaned.append(l)
    return "\n".join(cleaned)


def is_duplicate_question(new_q: str, past_qs: List[str]) -> bool:
    words_new = set(re.findall(r'\w+', new_q.lower()))
    if not words_new: return False
    
    for past_q in past_qs:
        words_old = set(re.findall(r'\w+', past_q.lower()))
        if not words_old: continue
        
        overlap = len(words_new.intersection(words_old)) / len(words_old)
        if overlap >= 0.6:  # 60% overlap in words counts as a duplicate
            return True
    return False

def generate_podcast_stream() -> Generator[str, None, None]:

    question_num = 1
    current_host_question = f"{INITIAL_QUESTION}"
    current_host_question_ = f"{question_num}. {INITIAL_QUESTION}"

    # 1. Host asks initial question (pre-rendered)
    yield f"data: {json.dumps({'speaker': 'host', 'is_new': True, 'chunk': current_host_question_})}\n\n"
    
    current_speaker = "guest"
    current_guest_answer = ""
    asked_questions = [INITIAL_QUESTION]
    past_topics = []
    
    for _ in range(TURNS - 1):
        if current_speaker == "guest":
            # Guest replies
            yield f"data: {json.dumps({'speaker': 'guest', 'is_new': True})}\n\n"
            
            parts = [f"Topic: {PODCAST_TOPIC}"]
            parts.append(f"{HOST_NAME} asks:\n{current_host_question}")
            user_prompt = "\n".join(parts)
            
            messages = [
                SystemMessage(content=GUEST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            
            full_response = ""
            for chunk in llm_dialogue.stream(messages):
                full_response += chunk.content
                for char in chunk.content:
                    yield f"data: {json.dumps({'speaker': 'guest', 'is_new': False, 'chunk': char})}\n\n"
                    time.sleep(TYPING_SPEED)
                
            current_guest_answer = clean_output(full_response.strip())
            current_speaker = "host"
            
        else:
            # Host replies
            yield f"data: {json.dumps({'speaker': 'host', 'is_new': True})}\n\n"
            
            question_num += 1
            raw_q, topic_data = run_host_pipeline(
                llm_dialogue, 
                PODCAST_TOPIC,
                current_guest_answer, 
                past_topics, 
                asked_questions
            )
            
            if topic_data and topic_data.get("topic") and topic_data.get("topic") != "Unknown":
                past_topics.append(topic_data["topic"])
                
            stripped_q = re.sub(r'^\d+\.\s*', '', raw_q.strip())
            current_host_question = f"{question_num}. {stripped_q}"
                
            # Stream artificially since we had to block on generation to validate
            for char in current_host_question:
                yield f"data: {json.dumps({'speaker': 'host', 'is_new': False, 'chunk': char})}\n\n"
                time.sleep(TYPING_SPEED)
                
            asked_questions.append(stripped_q)
            current_speaker = "guest"



# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html", topic=PODCAST_TOPIC)

@app.route("/stream")
def stream():
    return Response(generate_podcast_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
