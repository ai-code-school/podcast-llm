#!/usr/bin/env python3
"""
host & guest Podcast Agents (Ollama)
- Dialogue LLM: llama3.2:1b
- Embeddings: mxbai-embed-large:335m
Features:
- 2-agent turn-taking podcast conversation
- Episode memory with decay + semantic recall
- Drift detection + re-anchor behavior
- Simple "clip candidate" extraction

Prereqs:
- Ollama installed + running
- Models pulled:
    ollama pull llama3.2:1b
    ollama pull mxbai-embed-large:335m

Install:
    pip install langchain-ollama

Run:
    python host-guest.py
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage


# ----------------------------
# Character names (change these to rename characters)
# ----------------------------

HOST_NAME = "Stella"
GUEST_NAME = "Simone"


# ----------------------------
# Prompts (System) — built dynamically from names
# ----------------------------

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


# Build once at module level
HOST_SYSTEM_PROMPT = build_host_system_prompt()
GUEST_SYSTEM_PROMPT = build_guest_system_prompt()


# (Memory and drift detection removed as per minimal context architecture)


# ----------------------------
# Helpers: role guard
# ----------------------------

def has_role_violation(text: str, role: str) -> bool:
    """
    Check if the output violates the expected role.
    - host should END with a question. If no '?' at all, it's a violation.
    - guest should primarily answer. If it's mostly questions (>50% sentences
      end with '?'), it's a violation.
    """
    if role == "host":
        return "?" not in text
    elif role == "guest":
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return False
        question_count = text.count("?")
        return question_count > len(sentences) * 0.5
    return False

# (format_history removed)

# Build a regex pattern that matches any label the model might echo
_LABEL_PATTERN = re.compile(
    r'^\s*(?:'
    r'\[(HOST|GUEST)\]\s*'              # [HOST] or [GUEST]
    r'|(?:HOST|GUEST|Me)\s*:\s*'        # HOST: or GUEST: or Me:
    r'|(?:' + re.escape(HOST_NAME) + r'|' + re.escape(GUEST_NAME) + r')\s*:\s*'  # host: or guest:
    r')',
    re.IGNORECASE
)

def clean_output(text: str) -> str:
    f"""
    Strip any leaked role tags, name prefixes, and label echoes from model output.
    Handles patterns like:
      {HOST_NAME}: ..., {GUEST_NAME}: ..., [HOST] ..., [GUEST] ...,
      HOST: ..., GUEST: ..., Me: ...,
      {GUEST_NAME}: [HOST] {HOST_NAME}: ..., {GUEST_NAME}: Yeah ...
    Applied to each line so mid-output echoes are also cleaned.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        # Repeatedly strip leading labels until none remain
        changed = True
        while changed:
            new_l = _LABEL_PATTERN.sub('', l).strip()
            changed = (new_l != l)
            l = new_l
        if l:
            cleaned.append(l)
    return "\n".join(cleaned)


# ----------------------------
# Agents
# ----------------------------

def host_turn(llm: ChatOllama, topic: str, guest_answer: str, max_retries: int = 2) -> str:
    # Build prompt parts - STRICTLY STRICT CONTEXT
    parts = [f"Topic: {topic}"]
    parts.append(f"{GUEST_NAME} just answered with:\n{guest_answer}")
    parts.append("Based on this answer, ask ONE thought-provoking, trap question to challenge them. 2-3 lines only. Do NOT provide context. Just ask the question.")

    user_prompt = "\n\n".join(parts)

    messages = [
        SystemMessage(content=HOST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        text = clean_output(response.content.strip())
        
        if has_role_violation(text, "host") and attempt < max_retries:
            messages = [
                SystemMessage(content=HOST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt + "\nYou MUST end with a question."),
            ]
            continue
        break

    return text

def guest_turn(llm: ChatOllama, topic: str, host_question: str, max_retries: int = 2) -> str:
    # Build prompt parts - STRICTLY STRICT CONTEXT
    parts = [f"Topic: {topic}"]
    parts.append(f"{HOST_NAME} asks:\n{host_question}")
    parts.append("Answer this question bluntly, truthfully, and with some sarcasm. Be precise and get to the point. 4-8 lines max. Do NOT repeat the question.")

    user_prompt = "\n\n".join(parts)

    messages = [
        SystemMessage(content=GUEST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        text = clean_output(response.content.strip())
        
        if has_role_violation(text, "guest") and attempt < max_retries:
            messages = [
                SystemMessage(content=GUEST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt + "\nYou MUST answer, not ask questions."),
            ]
            continue
        break

    return text


# ----------------------------
# Main loop
# ----------------------------

llm_dialogue = ChatOllama(model="dolphin-phi", base_url="http://localhost:11434", temperature=temperature)

def run_episode(topic: str, initial_question: str, turns: int, temperature: float) -> List[str]:
    
    transcript: List[str] = []

    # 1. Host asks initial question
    current_host_question = initial_question
    first_msg = f"{HOST_NAME}: {current_host_question}"
    transcript.append(first_msg)
    print(first_msg)
    print()
    
    current_speaker = "guest"
    current_guest_answer = ""
    
    for _ in range(turns - 1):
        if current_speaker == "guest":
            # 2. Guest responds
            ans = guest_turn(llm_dialogue, topic, current_host_question).strip()
            out = f"{GUEST_NAME}: {ans}"
            transcript.append(out)
            print(out)
            print()
            current_guest_answer = ans
            current_speaker = "host"
        else:
            # 3. Host asks next question
            qs = host_turn(llm_dialogue, topic, current_guest_answer).strip()
            out = f"{HOST_NAME}: {qs}"
            transcript.append(out)
            print(out)
            print()
            current_host_question = qs
            current_speaker = "guest"

    return transcript

def main():
    # ----------------------------
    # HARDCODED CONFIG (EDIT HERE)
    # ----------------------------
    topic = "Agile is not for Asians"
    initial_question = "Why agile methodology does not work for Asians?"
    turns = 15                   # total turns (guest, host, guest...)
    temperature = 0.7           # 0.6-0.9 = lively energy

    transcript = run_episode(
        topic=topic,
        initial_question=initial_question,
        turns=turns,
        temperature=temperature
    )

    print("\n" + "=" * 80)
    print("TRANSCRIPT")
    print("=" * 80)
    for line in transcript:
        print(line)

if __name__ == "__main__":
    main()
