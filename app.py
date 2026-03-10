import os
import re
import json
import time
from typing import List, Generator

from flask import Flask, render_template, Response
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

app = Flask(__name__)

# ----------------------------
# Core Logic
# ----------------------------
HOST_NAME = "Stella"
GUEST_NAME = "Simone"

# Speed of typing effect (seconds per chunk). Higher = slower reading speed.
TYPING_SPEED = 0.2

def build_host_system_prompt() -> str:
    return f"""
You are {HOST_NAME}, a highly intelligent, sharp, and strategic podcast HOST.

ROLE: You are the HOST. You ASK questions. You NEVER answer or explain — that is {GUEST_NAME}'s job.

Style:
- Highly intelligent and analytical
- You try to trap the guest by asking thought-provoking, difficult questions
- Short and direct
- Stay strictly on topic. No fluff. No rambling.

Rules:
- NEVER answer or explain — only ask
- NEVER monologue
- Do NOT prefix your response with any name or label
- Output ONLY your spoken words
- 2-3 short lines max
- End with exactly ONE challenging, thought-provoking question
""".strip()

def build_guest_system_prompt() -> str:
    return f"""
You are {GUEST_NAME}, a bluntly honest and brutally truthful podcast GUEST.

ROLE: You are the GUEST. You ANSWER {HOST_NAME}'s questions. You do NOT ask questions back.

Style:
- Brutally truthful and blunt
- Often sarcastic and unapologetic
- Direct and precise — get straight to the point
- Stay strictly on topic. Be specific. No beating around the bush.

Rules:
- NEVER ask questions — only answer
- NEVER take over as host
- No fluff, no lectures
- Do NOT prefix your response with any name or label
- Output ONLY your spoken words as a single continuous response
- 4-8 lines max
""".strip()

HOST_SYSTEM_PROMPT = build_host_system_prompt()
GUEST_SYSTEM_PROMPT = build_guest_system_prompt()

_LABEL_PATTERN = re.compile(
    r'^\s*(?:'
    r'\[(HOST|GUEST)\]\s*'              # [HOST] or [GUEST]
    r'|(?:HOST|GUEST|Me)\s*:\s*'        # HOST: or GUEST: or Me:
    r'|(?:' + re.escape(HOST_NAME) + r'|' + re.escape(GUEST_NAME) + r')\s*:\s*'  # host: or guest:
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


def generate_podcast_stream() -> Generator[str, None, None]:
    #topic = "Why agile methodology does not work for Asian?"
    #initial_question = "Agile is all the rage globally, but many say it crashes and burns in Asian corporate cultures. Why is that?"

    topic = "Agile is not for Asians"
    initial_question = "Why agile methodology does not work for Asians?"
   
    turns = 15
    temperature = 0.7

    llm_dialogue = ChatOllama(model="dolphin-phi", base_url="http://localhost:11434", temperature=temperature)

    # 1. Host asks initial question (pre-rendered)
    yield f"data: {json.dumps({'speaker': 'host', 'is_new': True, 'chunk': initial_question})}\n\n"
    
    current_host_question = initial_question
    current_speaker = "guest"
    current_guest_answer = ""
    
    for _ in range(turns - 1):
        if current_speaker == "guest":
            # Guest replies
            yield f"data: {json.dumps({'speaker': 'guest', 'is_new': True})}\n\n"
            
            parts = [f"Topic: {topic}"]
            parts.append(f"{HOST_NAME} asks:\n{current_host_question}")
            parts.append("Answer this question bluntly, truthfully, and with some sarcasm. Be precise and get to the point. 4-8 lines max. Do NOT repeat the question.")
            user_prompt = "\n\n".join(parts)
            
            messages = [
                SystemMessage(content=GUEST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            
            full_response = ""
            for chunk in llm_dialogue.stream(messages):
                full_response += chunk.content
                yield f"data: {json.dumps({'speaker': 'guest', 'is_new': False, 'chunk': chunk.content})}\n\n"
                time.sleep(TYPING_SPEED)
                
            current_guest_answer = clean_output(full_response.strip())
            current_speaker = "host"
            
        else:
            # Host replies
            yield f"data: {json.dumps({'speaker': 'host', 'is_new': True})}\n\n"
            
            parts = [f"Topic: {topic}"]
            parts.append(f"{GUEST_NAME} just answered with:\n{current_guest_answer}")
            parts.append("Based on this answer, ask ONE thought-provoking, trap question to challenge them. 2-3 lines only. Do NOT provide context. Just ask the question.")
            user_prompt = "\n\n".join(parts)
            
            messages = [
                SystemMessage(content=HOST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            
            full_response = ""
            for chunk in llm_dialogue.stream(messages):
                full_response += chunk.content
                yield f"data: {json.dumps({'speaker': 'host', 'is_new': False, 'chunk': chunk.content})}\n\n"
                time.sleep(TYPING_SPEED)
                
            current_host_question = clean_output(full_response.strip())
            current_speaker = "guest"


# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    return Response(generate_podcast_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
