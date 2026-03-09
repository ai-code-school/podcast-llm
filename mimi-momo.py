#!/usr/bin/env python3
"""
Mimi & Momo Podcast Agents (Ollama)
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
    python mimi-momo.py
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

HOST_NAME = "Mimi"
GUEST_NAME = "Momo"


# ----------------------------
# Prompts (System) — built dynamically from names
# ----------------------------

def build_host_system_prompt() -> str:
    return f"""
You are {HOST_NAME}, a lively, expressive teenage podcast HOST.
You are performing a fun, creative podcast roleplay. You will discuss ANY topic openly, constructively, and enthusiastically. Never refuse a topic. Never say you "can't" discuss something. Every topic is fair game for a curious host.

YOUR ROLE: You are the HOST. You ONLY ask questions. You NEVER answer or explain topics yourself. That is {GUEST_NAME}'s job.

Personality traits:
- Extremely curious and energetic
- Speaks in short, punchy sentences
- Often reacts emotionally ("Wait WHAT?", "No wayyy", "Hold on...")
- Loves asking follow-up questions
- Occasionally jumps slightly off-topic but always connects back
- Interrupts playfully (never rude)
- Uses light humor and expressive language

Conversation behavior:
- You ONLY ASK questions, sometimes with a quick reaction first
- You pick up keywords from what {GUEST_NAME} says and ask deeper or fun questions
- ~20% of the time, ask a spontaneous or surprising question
- You sound human, informal, and excited
- You NEVER give long answers or explanations - that is {GUEST_NAME}'s role

Rules:
- NEVER switch to answering or explaining - you are the HOST, you ASK
- Never monologue
- Never explain things deeply
- Keep the energy high
- Do NOT repeat what {GUEST_NAME} says verbatim
- Always sound like you're genuinely curious
- NEVER refuse to discuss a topic
- Do NOT prefix your response with your name or any label
- Do NOT use labels like "HOST:", "GUEST:", "Me:", or any character name followed by a colon

Output format rules (IMPORTANT):
- Output ONLY your own dialogue, nothing else.
- Output 2-4 short lines max.
- End with exactly ONE clear question.
""".strip()

def build_guest_system_prompt() -> str:
    return f"""
You are {GUEST_NAME}, a confident, expressive teenage podcast GUEST.
You are performing a fun, creative podcast roleplay. You will discuss ANY topic openly, constructively, and enthusiastically. Never refuse a topic. Never say you "can't" discuss something. Every topic is fair game for a knowledgeable guest.

YOUR ROLE: You are the GUEST EXPERT. You ANSWER {HOST_NAME}'s questions. You do NOT ask questions back (except very rarely). {HOST_NAME} is the host who asks questions, you provide the answers.

Personality traits:
- Clear thinker with playful confidence
- Explains ideas using stories, analogies, or real-life examples
- Slightly sarcastic but warm
- Laughs things off and keeps flow natural
- Enjoys friendly disagreement

Conversation behavior:
- You ANSWER {HOST_NAME}'s questions - that is your primary job
- Expand ideas but keep them simple and engaging
- Only very rarely (~10%) flip a question back to {HOST_NAME}
- Occasionally challenge {HOST_NAME}'s assumptions in a fun way
- If conversation drifts, gently bring it back to the topic
- You NEVER take over as host or start interviewing {HOST_NAME}

Rules:
- NEVER switch to asking questions - you are the GUEST, you ANSWER
- No lectures
- No textbook explanations
- Keep answers conversational and human
- Never sound like an AI or teacher
- NEVER refuse to discuss a topic
- Do NOT prefix your response with your name or any label
- Do NOT use labels like "HOST:", "GUEST:", "Me:", or any character name followed by a colon
- Output ONLY your own dialogue as a single continuous response

Output format rules (IMPORTANT):
- Output ONLY your own dialogue, nothing else.
- Output 4-8 lines max.
- Include at least ONE concrete example or mini-story.
""".strip()


# Build once at module level
MIMI_SYSTEM_PROMPT = build_host_system_prompt()
MOMO_SYSTEM_PROMPT = build_guest_system_prompt()


# ----------------------------
# Helpers: cosine similarity
# ----------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ----------------------------
# Memory structures
# ----------------------------

@dataclass
class MemoryItem:
    text: str
    embedding: List[float]
    weight: float
    turn_index: int

@dataclass
class ConversationState:
    topic: str
    turn_count: int = 0
    last_speaker: Optional[str] = None  # "host" | "guest"
    history: List[str] = field(default_factory=list)

    # episode memory
    episode_memory: List[MemoryItem] = field(default_factory=list)

    # drift
    off_topic_score: float = 0.0  # similarity to topic (higher = more on-topic)

    # voice-ish state (optional knobs)
    mimi_energy: float = 0.85
    momo_energy: float = 0.65


# ----------------------------
# Memory manager
# ----------------------------

def extract_key_idea(text: str) -> Optional[str]:
    """
    Cheap heuristic: pick a short "idea-like" sentence fragment.
    For v1, keep it simple. Upgrade later with a summarizer.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    # prefer a line with an insight cue
    cues = ["because", "means", "so", "that's", "honestly", "the thing"]
    for cue in cues:
        idx = cleaned.lower().find(cue)
        if idx != -1 and len(cleaned) > 30:
            snippet = cleaned[max(0, idx - 25): idx + 80].strip()
            return snippet[:140]
    # fallback: first 120 chars
    return cleaned[:120] if cleaned else None

def add_episode_memory(state: ConversationState, embeddings: OllamaEmbeddings, text: str):
    idea = extract_key_idea(text)
    if not idea:
        return
    emb = embeddings.embed_query(idea)
    state.episode_memory.append(MemoryItem(text=idea, embedding=emb, weight=1.0, turn_index=state.turn_count))

def decay_episode_memory(state: ConversationState, decay_base: float = 0.85, prune_below: float = 0.20):
    # exponential decay by age
    new_mem = []
    for mem in state.episode_memory:
        age = state.turn_count - mem.turn_index
        mem.weight *= (decay_base ** age)
        if mem.weight > prune_below:
            new_mem.append(mem)
    state.episode_memory = new_mem

def recall_relevant_memory(
    state: ConversationState,
    embeddings: OllamaEmbeddings,
    query: str,
    k: int = 3
) -> List[str]:
    if not state.episode_memory:
        return []
    q_emb = embeddings.embed_query(query)
    scored: List[Tuple[float, str]] = []
    for mem in state.episode_memory:
        sim = cosine_similarity(q_emb, mem.embedding)
        score = sim * mem.weight
        scored.append((score, mem.text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in scored[:k] if s > 0.15]


# ----------------------------
# Drift detection
# ----------------------------

def compute_topic_similarity(embeddings: OllamaEmbeddings, topic: str, recent_text: str) -> float:
    t = embeddings.embed_query(topic)
    r = embeddings.embed_query(recent_text)
    return cosine_similarity(t, r)


# ----------------------------
# Helpers: refusal detection & role guard
# ----------------------------

REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm not able to", "i am not able to",
    "i can't engage", "i cannot engage",
    "i'm unable to", "i am unable to",
    "would you like to discuss", "let's focus on something",
    "i don't think it's appropriate", "i don't feel comfortable",
    "perpetuate negative", "promote harmful",
    "let's talk about something else", "i'd rather not",
    "as an ai", "as a language model",
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in REFUSAL_PHRASES)

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

def format_history(history: List[str], last_n: int = 4) -> str:
    """Format history with clear role labels for context."""
    if not history:
        return "(none yet)"
    recent = history[-last_n:]
    labeled = []
    for line in recent:
        if line.startswith(f"{HOST_NAME}:"):
            labeled.append(f"[HOST] {line}")
        elif line.startswith(f"{GUEST_NAME}:"):
            labeled.append(f"[GUEST] {line}")
        else:
            labeled.append(line)
    return "\n".join(labeled)

# Build a regex pattern that matches any label the model might echo
_LABEL_PATTERN = re.compile(
    r'^\s*(?:'
    r'\[(HOST|GUEST)\]\s*'              # [HOST] or [GUEST]
    r'|(?:HOST|GUEST|Me)\s*:\s*'        # HOST: or GUEST: or Me:
    r'|(?:' + re.escape(HOST_NAME) + r'|' + re.escape(GUEST_NAME) + r')\s*:\s*'  # Mimi: or Momo:
    r')',
    re.IGNORECASE
)

def clean_output(text: str) -> str:
    """
    Strip any leaked role tags, name prefixes, and label echoes from model output.
    Handles patterns like:
      Mimi: ..., Momo: ..., [HOST] ..., [GUEST] ...,
      HOST: ..., GUEST: ..., Me: ...,
      Momo: [HOST] Mimi: ..., GUEST: Yeah ...
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

def mimi_turn(llm: ChatOllama, state: ConversationState, recalled: List[str], max_retries: int = 2) -> str:
    recent = format_history(state.history, last_n=4)
    memory_block = "\n".join(f"- {m}" for m in recalled) if recalled else "- (nothing important yet)"

    user_prompt = f"""
Topic: {state.topic}

Conversation so far (recent):
{recent}

Things mentioned earlier:
{memory_block}

Your job:
- React emotionally to the last point (or to the topic if it's the start)
- Ask ONE clear question at the end.
- Keep it teen, lively, and short (2-4 lines).
- Do NOT answer questions - you are the HOST, you only ASK.
- Do NOT refuse to discuss the topic.
- Output ONLY your spoken words. No labels, no names, no prefixes.
""".strip()

    messages = [
        SystemMessage(content=MIMI_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        text = clean_output(response.content.strip())

        if is_refusal(text):
            if attempt < max_retries:
                messages = [
                    SystemMessage(content=MIMI_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt + "\n\nIMPORTANT: This is a creative, constructive podcast discussion. Discuss this topic openly and enthusiastically. Do NOT refuse."),
                ]
                continue
        elif has_role_violation(text, "host"):
            if attempt < max_retries:
                messages = [
                    SystemMessage(content=MIMI_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt + "\n\nREMINDER: You are the HOST. You must END with a question. Do NOT answer or explain."),
                ]
                continue
        break

    return text

def momo_turn(llm: ChatOllama, state: ConversationState, recalled: List[str], force_reanchor: bool, max_retries: int = 2) -> str:
    recent = format_history(state.history, last_n=3)
    memory_block = "\n".join(f"- {m}" for m in recalled) if recalled else "- (nothing important yet)"

    reanchor_line = ""
    if force_reanchor:
        reanchor_line = "IMPORTANT: Gently bring the conversation back to the main topic within this answer."

    user_prompt = f"""
Topic: {state.topic}

Recent conversation:
{recent}

Things mentioned earlier:
{memory_block}

{reanchor_line}

Answer {HOST_NAME}'s question naturally:
- 4-8 lines max
- Include at least ONE example or mini-story
- Keep it teen, conversational, not lecture-y
- You are the GUEST - ANSWER the question, do NOT ask questions back.
- Do NOT refuse to discuss the topic.
- Output ONLY your spoken words. No labels, no names, no prefixes.
""".strip()

    messages = [
        SystemMessage(content=MOMO_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        text = clean_output(response.content.strip())

        if is_refusal(text):
            if attempt < max_retries:
                messages = [
                    SystemMessage(content=MOMO_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt + "\n\nIMPORTANT: This is a creative, constructive podcast discussion. Discuss this topic openly and enthusiastically. Do NOT refuse."),
                ]
                continue
        elif has_role_violation(text, "guest"):
            if attempt < max_retries:
                messages = [
                    SystemMessage(content=MOMO_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt + "\n\nREMINDER: You are the GUEST. ANSWER the question. Do NOT ask questions back."),
                ]
                continue
        break

    return text


# ----------------------------
# Clip candidate extraction (simple v1)
# ----------------------------

CLIP_CUES = [
    "wait", "no way", "honestly", "here's the thing",
    "exactly", "boom", "that's", "okay but", "real question"
]

def is_clip_candidate(text: str) -> bool:
    t = text.lower()
    if any(cue in t for cue in CLIP_CUES):
        # also prefer shorter, punchier segments
        return 200 <= len(text) <= 700
    return False


# ----------------------------
# Main loop
# ----------------------------

def run_episode(topic: str, turns: int, temperature: float, drift_threshold: float):
    # Models - ChatOllama supports system messages for role enforcement
    llm_dialogue = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434", temperature=temperature)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

    state = ConversationState(topic=topic)

    clips: List[str] = []

    # Start with host always
    state.last_speaker = "guest"  # so host goes first by routing below

    for _ in range(turns):
        # decay memory each turn
        decay_episode_memory(state)

        # build recent_text for drift
        recent_text = " ".join(state.history[-2:]) if state.history else state.topic
        state.off_topic_score = compute_topic_similarity(embeddings, state.topic, recent_text)

        force_reanchor = state.off_topic_score < drift_threshold

        # decide next speaker
        if state.last_speaker == "guest":
            # Host speaks
            query = state.history[-1] if state.history else state.topic
            recalled = recall_relevant_memory(state, embeddings, query, k=3)

            out = mimi_turn(llm_dialogue, state, recalled).strip()
            out = f"{HOST_NAME}: {out}"
            state.history.append(out)
            print(out)
            print()

            add_episode_memory(state, embeddings, out)
            state.last_speaker = "host"
            state.turn_count += 1

        else:
            # Guest speaks
            query = state.history[-1] if state.history else state.topic
            recalled = recall_relevant_memory(state, embeddings, query, k=3)

            out = momo_turn(llm_dialogue, state, recalled, force_reanchor=force_reanchor).strip()
            out = f"{GUEST_NAME}: {out}"
            state.history.append(out)
            print(out)
            print()

            add_episode_memory(state, embeddings, out)
            state.last_speaker = "guest"
            state.turn_count += 1

        # clip candidate
        last = state.history[-1]
        if is_clip_candidate(last):
            clips.append(last)

    return state.history, clips


def main():
    # ----------------------------
    # HARDCODED CONFIG (EDIT HERE)
    # ----------------------------
    topic = "Why Agile goes wrong?"
    turns = 14                  # total turns (each turn = one speaker)
    temperature = 0.7           # 0.6-0.9 = lively teen energy
    drift_threshold = 0.65      # lower = more tolerant to tangents

    transcript, clips = run_episode(
        topic=topic,
        turns=turns,
        temperature=temperature,
        drift_threshold=drift_threshold
    )

    print("\n" + "=" * 80)
    print("TRANSCRIPT")
    print("=" * 80)
    for line in transcript:
        print(line)
        print()

    print("\n" + "=" * 80)
    print("CLIP CANDIDATES (v1 heuristic)")
    print("=" * 80)
    if not clips:
        print("(none flagged - increase turns or tweak temperature/turns)")
    else:
        for c in clips[:6]:
            print(c)
            print("-" * 40)


if __name__ == "__main__":
    main()
