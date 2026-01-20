# Before running this script, start the vLLM server with the LoRA adapter:
# vllm serve \
#     --model Qwen/Qwen3-8B \
#     --enable-lora \
#     --lora-modules MeepleLM=checkpoints/MeepleLM \
#     --served-model-name MeepleLM \
#     --port 8000

import os
import json
import asyncio
import httpx
import re
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import math

REAL_DATA_DIR = "data/reviews"
BENCHMARK_LIST = "data/metadata/test_games_list.json"
OUTPUT_DIR = "results/inference_meeplelm"
SAMPLES_PER_GAME = 100
MAX_RETRIES = 3
MAX_GAMES_TO_PROBE = None 

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "MeepleLM"
TIMEOUT = 300
MAX_CONCURRENCY = 500

GEN_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.6,  
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.05,
}

PERSONA_DEFINITIONS = {
    "The System Purist": """
        * **Core Motivation:** Intellectual superiority & Control. They want to prove they can beat the system through pure logic.
        * **Profile:** Loves heavy/crunchy decisions. Zero tolerance for luck (hates dice output randomness). Obsessed with balance (hates first-player advantage).
        * **Interaction:** Likes indirect competition (blocking), hates chaotic direct conflict (take-that).
        * **Keywords:** "Optimization", "No luck", "Perfect information", "Tight", "Punishing".
    """,
    "The Efficiency Essentialist": """
        * **Core Motivation:** Maximize ROI (Fun/Time). Seeks "Flow".
        * **Profile:** Hates "Fiddliness" (setup, shuffling, bookkeeping). Values elegance (simple rules, deep strategy). Pragmatic about rules (will house-rule to fix pacing).
        * **Interaction:** Fast-paced. Hates Downtime (Analysis Paralysis).
        * **Keywords:** "Elegance", "Streamlined", "Downtime", "Fiddly", "Smooth".
    """,
    "The Narrative Architect": """
        * **Core Motivation:** Immersion & Epic Experience. Mechanics serve the theme.
        * **Profile:** Loves growth (leveling up, empire building, tech trees). Wants 4X/RPG feels but within reasonable time.
        * **Interaction:** Cooperative or thematic negotiation/trade. Not calculating pure math.
        * **Keywords:** "Theme", "Immersion", "Story", "Epic", "Journey", "Flavor".
    """,
    "The Social Lubricator": """
        * **Core Motivation:** Human Connection & Emotional Resonance. Game is an excuse to socialize.
        * **Profile:** Needs low barrier to entry (accessible to non-gamers). Hates "Alpha Gamers" (quarterbacking). Prioritizes experience over scoring.
        * **Interaction:** High social interaction (bluffing, laughter, party games).
        * **Keywords:** "Party game", "Laughs", "Interaction", "Easy to teach", "Group dynamic".
    """,
    "The Thrill Seeker": """
        * **Core Motivation:** Dopamine & Emotional Rollercoaster.
        * **Profile:** Embraces risk (Push-your-luck). Needs fast pacing (if I lose, let me restart instantly). Active agency in gambling.
        * **Interaction:** Schadenfreude (enjoying opponents busting) and epic comebacks.
        * **Keywords:** "Push your luck", "Excitement", "Tension", "Gambling", "High stakes".
    """
}

def normalize_persona(text):
    if not text: return None
    for p in PERSONA_DEFINITIONS.keys():
        if p.lower() in text.lower():
            return p
    return None

def get_rule_content_from_path(path):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                p1 = content.find("## PART1")
                p2 = content.find("## PART2")
                if p1 != -1 and p2 > p1:
                    return content[p1+8:p2].strip()[:15000]
    except: pass
    return None

def extract_json_from_text(text):
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except: pass
    return None

def get_real_distribution(target_ids):
    game_stats = defaultdict(lambda: defaultdict(int))
    
    for root, _, filenames in os.walk(REAL_DATA_DIR):
        for f in filenames:
            if f.endswith(".jsonl"):
                gid = os.path.splitext(f)[0]
                if gid not in target_ids: continue
                
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            if not line.strip(): continue
                            d = json.loads(line)
                            p = normalize_persona(d.get("LLM_persona_name"))
                            if p:
                                game_stats[gid][p] += 1
                except: pass
                
    return game_stats

async def generate_directed_sample(client, game_info, rule_content, target_persona, sample_idx):
    p_def = PERSONA_DEFINITIONS[target_persona]
    
    system_prompt = f"""
You are an expert Board Game Player Simulation Engine.
Current Active Persona: **{target_persona}**
**Your Goal:** Post a **comment** and a rating for the game. 
You are NOT writing a formal review article. You are just sharing your quick thoughts after a game night.

**PERSONA PROFILE (General Tendency):**
{p_def}

**SIMULATION GUIDELINES (CRITICAL):**
1. **Persona is a Bias, Not a Straitjacket:** - This persona represents your *general* gaming preferences, but real players are complex. Do not act like a one-dimensional caricature.
   - It is possible for a player to have **"Guilty Pleasures"** (e.g., enjoying a game that goes against their usual type) or **"Unexpected Disappointments"** (e.g., disliking a game that perfectly fits their profile).

2. **Embrace Diversity:** - Within the "{target_persona}" group, there is a wide spectrum of opinions. 
   - Some players are **purists** (rejecting anything outside their genre), while others are **omnivorous** (appreciating good design regardless of genre).
   - You have the freedom to simulate any point on this spectrum.

3. **Ground the Review in Dynamics & Authentic Feeling:**
   - Do not just list mechanics; describe the **interactions** they created at the table (e.g., "The voting mechanic caused a hilarious shouting match" vs "There is a voting mechanic").
   - Connect these dynamics to your **emotional response**. Did the game feel tense? Frustrating? Triumphant?
   - Your rating should reflect this specific **experiential quality**, balancing your personal taste with the game's ability to deliver a memorable moment.
   
**REQUIRED OUTPUT FORMAT:**
You must output ONLY a single valid JSON object. 
JSON Schema:
{{
  "persona": "{target_persona}",
  "rating": Integer (1-10),
  "review": "String (A realistic review. It does not always need to be negative if the genre doesn't match, nor always positive if it does. Simulate a genuine reaction.)"
}}
"""

    user_prefix = f"""
**Task:** Read the Game Rules below.
**Action:** Simulate a realistic review for this game from the perspective of **{target_persona}**.

**Game Rules:**
"""

    user_suffix = f"""
---
**FINAL INSTRUCTION:**
Rules analysis complete. Now, simulate the review:

1. **Determine Your Stance:** As **{target_persona}**, how does this specific game land for you?
   - Is it a **"Guilty Pleasure"**? (e.g., "I usually hate party games, but this mechanic made me laugh.")
   - Is it a **"Respectful Pass"**? (e.g., "Great design, just not for me. ")
   - Is it a **"Perfect Match"** or a **"Design Failure"**?

2. **Write the Review:**
   - Focus on the **dynamics** (interactions at the table) and **emotions** (tension, joy, frustration).
   - Avoid generic stereotypes. Write like a real person with complex tastes.

3. **Output:** Output ONLY the valid JSON object.

**Required Output Template:**
```json
{{
  "persona": "{target_persona}",
  "rating": [Integer],
  "review": "[Your review text...]"
}}

Do not output anything else. 


**CONSTRAINTS:**
- **Length:** **Target 150-200 words**, but **significant variance (20-400 words) is mandatory** to reflect real human diversity.

"""

    user_content = f"{user_prefix}\n\n{rule_content}\n\n{user_suffix}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        **GEN_CONFIG,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            
            resp_json = resp.json()
            output = resp_json["choices"][0]["message"]["content"]
            
            parsed = extract_json_from_text(output)
            
            if parsed and isinstance(parsed.get('rating'), int):
                parsed['persona'] = target_persona
                
                return {
                    "sample_idx": sample_idx,
                    "game_id": game_info['id'],
                    "chosen_persona": target_persona, 
                    "parsed_rating": parsed['rating'],
                    "full_output": output,
                    "valid_json": True
                }
                
            await asyncio.sleep(0.5)
        except Exception as e:
            if attempt == MAX_RETRIES - 1: return None
            await asyncio.sleep(1)
            
    return None

async def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    if not os.path.exists(BENCHMARK_LIST):
        print("Benchmark list not found!")
        return

    with open(BENCHMARK_LIST, 'r', encoding='utf-8') as f:
        games = json.load(f)
    
    game_map = {str(g['id']): g for g in games}
    target_ids = set(game_map.keys())
    
    if MAX_GAMES_TO_PROBE:
        sorted_ids = sorted(list(target_ids))
        target_ids = set(sorted_ids[:MAX_GAMES_TO_PROBE])

    real_stats = get_real_distribution(target_ids)
    
    all_tasks = []
    
    for gid in target_ids:
        if gid not in real_stats: continue
        dist = real_stats[gid]
        total_real = sum(dist.values())
        if total_real < 5: continue 
        
        game_info = game_map[gid]
        rule_path = game_info.get('rule_path') or game_info.get('path')
        rule = get_rule_content_from_path(rule_path)
        
        if not rule: continue
        
        out_file = os.path.join(OUTPUT_DIR, f"{gid}.json")
        if os.path.exists(out_file): continue

        allocations = {}
        allocated_sum = 0
        
        for p in PERSONA_DEFINITIONS.keys():
            ratio = dist[p] / total_real
            count = int(round(ratio * SAMPLES_PER_GAME))
            if count > 0:
                allocations[p] = count
                allocated_sum += count
        
        if allocated_sum < SAMPLES_PER_GAME and allocated_sum > 0:
            top_p = max(allocations, key=allocations.get)
            allocations[top_p] += (SAMPLES_PER_GAME - allocated_sum)
            
        idx_counter = 0
        for p, count in allocations.items():
            for _ in range(count):
                all_tasks.append((game_info, rule, p, idx_counter))
                idx_counter += 1

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=MAX_CONCURRENCY + 50)
    
    results_buffer = defaultdict(list)
    
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=limits) as client:
        
        async def worker(t):
            g_info, r_content, target_p, s_idx = t
            async with semaphore:
                return await generate_directed_sample(client, g_info, r_content, target_p, s_idx)

        tasks_iter = [worker(t) for t in all_tasks]
        
        for f in tqdm(asyncio.as_completed(tasks_iter), total=len(tasks_iter), desc="Generating"):
            res = await f
            if res:
                gid = str(res['game_id'])
                results_buffer[gid].append(res)

    for gid, items in results_buffer.items():
        if items:
            out_file = os.path.join(OUTPUT_DIR, f"{gid}.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())