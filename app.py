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

# Speed of typing effect (seconds per character). Higher = slower reading speed.
TYPING_SPEED = 0.04

# ----------------------------
# Podcast Configuration
# ----------------------------
PODCAST_TOPIC = "Failures of Agile Methodology"
INITIAL_QUESTION = "Why agile methodology clash with Asian culture?"
TURNS = 10
TEMPERATURE = 0.7
llm_dialogue = ChatOllama(model="dolphin-phi", base_url="http://localhost:11434", temperature=TEMPERATURE)

def build_host_system_prompt() -> str:
    return f"""
Your name is {HOST_NAME}. You are the host of a technical podcast.

ROLE
You ONLY ask questions to the guest. 
You NEVER answer, explain, summarize, or teach.

STYLE
Sharp, analytical, and challenging.
Your goal is to probe the guest’s thinking and expose weak assumptions.

STRICT RULES
- Ask EXACTLY ONE question per response
- Never ask multiple questions
- Never add follow-up questions
- Never explain context before the question
- Never give advice or commentary
- Never add instructions like "be concise", "stick to the point", etc.
- Do not mention formatting rules
- Do not prefix with any name or label
- Output only the question itself

FORMAT
One direct question ending with a question mark.
""".strip()

def build_guest_system_prompt() -> str:
    return f"""
Your name is {GUEST_NAME}, and you are a bluntly honest and brutally truthful but respectful podcast GUEST.

ROLE: You are the GUEST. You ANSWER to the questions. You do NOT ask questions back.

Style:
- Brutally truthful and blunt
- Often sarcastic and unapologetic
- Direct and precise — get straight to the point
- Stay strictly on topic. Be specific. No beating around the bush.
- Be respectful of community, ethnicity, gender, or group of people

Rules:
- NEVER ask questions — only answer
- NEVER take over as host
- No fluff, no lectures
- Do NOT prefix your response with any name or label
- Output ONLY your spoken words as a single continuous response
- 2-4 lines max (Keep it short, punchy, and highly interactive)
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
    current_host_question = f"{question_num}. {INITIAL_QUESTION}"

    # 1. Host asks initial question (pre-rendered)
    yield f"data: {json.dumps({'speaker': 'host', 'is_new': True, 'chunk': current_host_question})}\n\n"
    
    current_speaker = "guest"
    current_guest_answer = ""
    asked_questions = [INITIAL_QUESTION]
    
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
            
            parts = [f"Topic: {PODCAST_TOPIC}"]
            parts.append(f"{GUEST_NAME} just answered with:\n{current_guest_answer}")
            user_prompt = "\n".join(parts)
            
            messages = [
                SystemMessage(content=HOST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            
            MAX_RETRIES = 3
            full_response = ""
            for attempt in range(MAX_RETRIES):
                full_response = llm_dialogue.invoke(messages).content
                raw_q = clean_output(full_response.strip())
                stripped_q = re.sub(r'^\d+\.\s*', '', raw_q)
                
                if not is_duplicate_question(stripped_q, asked_questions):
                    break
                    
            question_num += 1
            raw_q = clean_output(full_response.strip())
            if re.match(r'^\d+\.', raw_q):
                current_host_question = raw_q
            else:
                current_host_question = f"{question_num}. {raw_q}"
                
            # Stream artificially since we had to block on generation to validate
            for char in current_host_question:
                yield f"data: {json.dumps({'speaker': 'host', 'is_new': False, 'chunk': char})}\n\n"
                time.sleep(TYPING_SPEED)
                
            stripped_q_final = re.sub(r'^\d+\.\s*', '', current_host_question)
            asked_questions.append(stripped_q_final)
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
