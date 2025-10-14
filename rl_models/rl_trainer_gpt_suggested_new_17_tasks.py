# ✅ GRPO on Reasoning Gym (local-friendly)
# - Works on Mac (MPS), CUDA, or CPU
# - Policy in FP32 (stable); judge defaults to a small model
# - LLM-as-judge, robust numeric extraction + fallback
# - No generation_kwargs on GRPOTrainer; sampling via model.generation_config

import os, re, json, random, torch
from typing import List, Any
from datasets import Dataset
import reasoning_gym
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer
import pandas as pd
from fuzzywuzzy import fuzz
from colorama import Fore, Back, Style
import numpy as np
# step 2 - load the embedding model

# this is a different model that is the best for english and zero shot
sbert_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# load the struct jsons for the level 9 data
test_df = pd.read_parquet("/Users/tariromashongamhende/Local Files/ml_projects/satellite_slug/project_eikon//experiments/reinforcement_learning/training_data/cleaned_land_use_test_data_4_options.parquet.gzip")
test_df = test_df.sample(frac=1).reset_index(drop=True)

# ----------------------------
# Devices
# ----------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def exaggerate_reward_signal(reward):
    def logit_sharpen(x, tau=0.5):
        x = np.clip(x, 1e-8, 1-1e-8)
        z = np.log(x/(1-x)) / tau
        return 1 / (1 + np.exp(-z))
    def squeeze_below(x, pivot=0.8, gamma=3.0):
        x = np.clip(x, 0.0, 1.0)
        below = (x/pivot)**gamma * pivot     # γ↑ = stronger squeeze
        return np.where(x < pivot, below, x)

    # output_reward = logit_sharpen(reward) # this is the less harsh version
    output_reward = float(squeeze_below(reward)) # this is the more harsh version penalising r < 0.8

    return output_reward

def generate_narrative_summary(struct_json_dict):

    prefix_statement = "A location has the following characteristics:\n\n"

    area_size_in_km = round(struct_json_dict["summary"]["area_size_km_2"],3)
    area_size_narrative_str = f"It has an area size of {area_size_in_km} km2. "
    
    building_count = struct_json_dict["summary"]["building"]["count"]
    if float(building_count)>0:
        building_count_narrative = f"There are {building_count} buildings. "
    else:
        building_count_narrative = f"There are 0 buildings. "

    
    building_density = struct_json_dict["summary"]["building_density_km_2"]
    if building_density == None:
        building_density = 0

    if float(building_density)>0:
        building_density_narrative = f"The building density is {round(building_density,3)}. "
    else:
        building_density_narrative = f"The building density is 0. "

    
    leisure = struct_json_dict["summary"]["leisure"]
    if len(leisure)>4:
        leisure_narrative_container = list(leisure)
        leisure_narrative_str = "The location has the following leisure types: \n" + "\n".join(leisure_narrative_container)
    else:
        leisure_narrative_str = "The location has no areas dedicated to leisure activities. "
        
    natural = struct_json_dict["summary"]["natural"]
    if len(natural)>4:
        natural_narrative_container = list(natural)#.tolist()
        natural_narrative_str = "The location has the following natural features: \n" + "\n".join(natural_narrative_container)
    else:
        natural_narrative_str = "The location has no noticeable natural features. "
        
    roads = struct_json_dict["summary"]["roads"]
    if len(roads)>4:
        roads_narrative_container = []
        road_keys = roads.keys()
        for road_type in road_keys:
            if float(roads[road_type])>0:
                road_type_narrative_str = f"There are {roads[road_type]} {road_type} roads. "
                roads_narrative_container.append(road_type_narrative_str)
            else:
                pass
        if len(roads_narrative_container)>0:
            road_type_narrative_prefix = "\n\nThe location has the following road type breakdown: \n"
            roads_narrative_str = road_type_narrative_prefix + "\n".join(roads_narrative_container)
        else:
            roads_narrative_str = "There are no identifiable roads. "
        
    sport = struct_json_dict["summary"]["sport"]
    if len(sport)>0:
        sports_narrative_container = []
        sports_keys = list(sport)
        for sport_type in sports_keys:
            sport_type_narrative_str = f"There is an area dedicated to {sport_type}."
            sports_narrative_container.append(sport_type_narrative_str)
        if len(sports_narrative_container)>0:
            sports_type_narrative_prefix = "\n\nThe location has the following areas for different sports: \n"
            sports_narrative_str = sports_type_narrative_prefix + "\n".join(sports_narrative_container)
        else:
            sports_narrative_str = "There are no areas of land reserved to any sporting activities. "
    else:
        sports_narrative_str = "There are no areas of land reserved to any sporting activities. "
    prefix_question = prefix_statement + area_size_narrative_str+ building_count_narrative + building_density_narrative + leisure_narrative_str + natural_narrative_str + roads_narrative_str + sports_narrative_str  
    suffix_question = "<question> What is the land use of this location? </question>"
    combined_question = prefix_question + "\n\n" +  suffix_question
    return combined_question

def preprocess_landuse_answer(answer_list):
    narrative_output_container = []
    for i in list(answer_list):
        narrative_output_container.append(i)
    narrative_output_str = "<answer> " + ", ".join(narrative_output_container) + " </answer>"
    narrative_output_str = ", ".join(narrative_output_container)
    return narrative_output_str

# make a regex extraction function for easily evaluating answers
def extract_answer(response):
    # simple regex search to extract answers
    answer = re.search(f"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer is not None:
        return answer.group(1)
    return answer
# make a regex extraction function for easily evaluating answers
def extract_question(response):
    # simple regex search to extract answers
    question = re.search(f"<question>(.*?)</question>", response, re.DOTALL)
    if question is not None:
        return question.group(1)
    return question

DEVICE = pick_device()
print(f"[Device] policy device = {DEVICE}")

# ----------------------------
# Config
# ----------------------------

MODEL_NAME  = os.getenv("MODEL_NAME", "/Users/tariromashongamhende/Local Files/ml_projects/satellite_slug/project_eikon/experiments/reinforcement_learning/sft_smollm_135b_for_reasoning")   # policy
# Use a small default judge so it runs locally. Override with env if you want.
# JUDGE_MODEL = os.getenv("JUDGE_MODEL", "HuggingFaceTB/SmolLM-1.7B")

TASK_NAME   = os.getenv("RG_TASK", "leg_counting")
TASK_NAME   = os.getenv("RG_TASK", "propositional_logic")

SIZE        = int(os.getenv("RG_SIZE", "50"))
SEED        = int(os.getenv("SEED", "42"))
OUT_DIR = f"/Users/tariromashongamhende/Local Files/ml_projects/satellite_slug/project_eikon/experiments/reinforcement_learning/eikon_reasoning_model_17_tasks"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED)

PROMPT_TMPL = (
    "You are a careful reasoner.\n"
    "Question: {q}\n"
    # "Answer with the final answer only.\n"
)

# ----------------------------
# Dataset
# ----------------------------
# rg_data = reasoning_gym.create_dataset(TASK_NAME, size=SIZE, seed=SEED)


list_of_tasks_from_reasoning_gym = ["aiw",
                                    "basic_arithmetic",
                                    "chain_sum",
                                    "color_cube_rotation",
                                    "countdown",
                                    "family_relationships",
                                    # "gsm_symbolic",
                                    "knights_knaves",
                                    "leg_counting",
                                    "letter_counting",
                                    "needle_haystack",
                                    "number_sorting",
                                    "propositional_logic",
                                    "self_reference",
                                    "sentence_reordering",
                                    "spell_backward",
                                    "syllogism",
                                    "zebra_puzzles"
                                    ]

import reasoning_gym
from datasets import Dataset





SIZE       = int(os.getenv("RG_SIZE", "10"))       # how many Q&A pairs to generate
SEED       = int(os.getenv("SEED", "42"))


INSTRUCT_PREFIX = (
    "You are a careful reasoner.\n"
    "Question: {q}\n"
    "Answer with the final answer only.\n"
)

INSTRUCT_PREFIX = ("""
You are a careful reasoner.

Generate an answer after thinking.
Your output format should be as follows:

**<think> you reasons here </think><answer> Answer here </answer>**
You must answer within the <answer>...</answer> !
Your response MUST NEVER be more than 100 words.

Question: {q}
Respond:

"""
)

answer_format_str = """Generate an answer after thinking.
Use <think> you reasons here </think><answer> Answer here </answer>
You must answer within the <answer>...</answer> !
Your response MUST NEVER be more than 100 words.

Example output:

<think>
<thought>To find the total number of legs, you need to calculate the legs of each animal and then add them together.</thought>
<thought>Spiders have 8 legs each and dogs have 4 legs each.</thought>
<thought>So, for three spiders: 3 * 8 = 24 legs</thought>
<thought>For three dogs: 3 * 4 = 12 legs</thought>
<thought>Now, add the legs of spiders and dogs together: 24 + 12 = 36</thought>
</think>

<answer> Therefore, there are 36 legs in total. </answer>

"""
PROMPT_TMPL = (
    "You are a careful reasoner.\n"
      f"{answer_format_str}\n\n"
      "Actual question below\n"
      "----------------------\n\n"

    "Question: {q}\n"
    # "Answer with the final answer only.\n"
)

rows = []
for task in list_of_tasks_from_reasoning_gym:
  print("-"*10)
  print(f"TASK: {task}")
  print("\n")
  rg_data = reasoning_gym.create_dataset(task, size=SIZE, seed=SEED)

  for entry in rg_data:
      q = entry["question"]
      gt = entry["answer"]
      formatted_entry ={
          # TRL expects a 'prompt' column in standard format
          "prompt": str(PROMPT_TMPL.format(q=q)),
          # keep the ground truth & full entry for reward function use
          "ground_truth": str(gt),
          "entry": str(entry),                # the raw RG entry used by the verifier
          "metadata": str(entry.get("metadata", {})),
      }

      rows.append(formatted_entry)
      print(type(formatted_entry))

print("-"*10)
print("DONE")
hf_ds = Dataset.from_list(rows)
print(len(hf_ds))
print(hf_ds[:5])

# rows = []
# for i in range(len(test_df)):
#     inscope_df = test_df[test_df.index==i].reset_index().drop(columns="index")
#     # check to make sure there is in fact a land use to predict 
#     # inscope_df["landuse_cat_count"] = inscope_df['osm_structured_json_dict'].apply(lambda x: len(x['summary']['landuse']))
#     # if inscope_df["landuse_cat_count"].values[0]>0:
#         # narrative_question_and_answers = generate_narrative_summary(inscope_df["osm_structured_json_dict"].values[0])
#         # entry_w_no_answer = narrative_question_and_answers.split("</question>")[0].replace("<question>","")
#         # narrative_answer = preprocess_landuse_answer(inscope_df["osm_structured_json_dict"].values[0]['summary']['landuse'])

#     narrative_question_and_answers = inscope_df["question"].values[0] + inscope_df["correct_answer"].values[0]
#     narrative_answer = inscope_df["correct_answer"].values[0]
#     entry_w_no_answer = inscope_df["question"].values[0]
    
#     rows.append({
#         "prompt": PROMPT_TMPL.format(q=(narrative_question_and_answers)),
#         "ground_truth": str(narrative_answer),
#         "entry": entry_w_no_answer,
#     })
#     # else:
#     #     pass
# hf_ds = Dataset.from_list(rows)
# print(len(hf_ds))
# print(hf_ds[0]["prompt"])
# print(hf_ds[0]["ground_truth"])
# print(hf_ds[0]["entry"])

print("**"*10)
# ----------------------------
# Tokenizer + Policy (FP32)
# ----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

policy = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,              # FP32 (stable across devices)
)
policy.to(DEVICE)
policy.config.use_cache = False
policy.gradient_checkpointing_enable()

# Stochastic generation (critical for GRPO reward variance)
gen_cfg = GenerationConfig.from_model_config(policy.config)
gen_cfg.do_sample   = True
gen_cfg.temperature = 0.9
gen_cfg.top_p       = 0.95
gen_cfg.top_k       = 50
policy.generation_config = gen_cfg

# ----------------------------
# LLM-as-judge (robust, no strict JSON)
# ----------------------------
# _JTOK = AutoTokenizer.from_pretrained(JUDGE_MODEL)
# if _JTOK.pad_token is None:
#     _JTOK.pad_token = _JTOK.eos_token
# _JTOK.padding_side = "left"

# # Keep judge on CPU by default to avoid VRAM fights; set JUDGE_DEVICE=mps/cuda to move it
# JUDGE_DEVICE = os.getenv("JUDGE_DEVICE", "cpu").lower()
# if JUDGE_DEVICE not in ("cpu", "cuda", "mps"):
#     JUDGE_DEVICE = "cpu"
# print(f"[Device] judge device = {JUDGE_DEVICE}")

# # Use fp16 only if GPU and you want speed; fp32 is safest
# _judge_dtype = torch.float16 if JUDGE_DEVICE in ("cuda", "mps") else torch.float32

# JUDGE = AutoModelForCausalLM.from_pretrained(
#     JUDGE_MODEL,
#     dtype=_judge_dtype,
# )
# JUDGE.to(JUDGE_DEVICE)
# JUDGE.config.use_cache = True

_NUM_RE = re.compile(r"-?\d+")
def _last_int(s: str):
    m = _NUM_RE.findall(s)
    return int(m[-1]) if m else None

_NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7",
    "eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12","thirteen":"13","fourteen":"14",
    "fifteen":"15","sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20"
}

def _normalize_number_words(s: str) -> str:
    return re.sub(r"\b(" + "|".join(_NUM_WORDS.keys()) + r")\b",
                  lambda m: _NUM_WORDS[m.group(1).lower()], s, flags=re.IGNORECASE)

def _extract_candidate_number_strict(candidate: str):
    # Convert simple number words to digits, then take the LAST integer in the candidate only
    s = _normalize_number_words(candidate.strip())
    ints = _NUM_RE.findall(s)
    return int(ints[-1]) if ints else None



# def judge_compare(candidate: str) -> int:
#     """
#     Ask the judge to extract ONE integer from the CANDIDATE_ANSWER only.
#     Returns that integer or None.
#     """
#     prompt = (
#         "Extract the FINAL numeric answer from the CANDIDATE_ANSWER below.\n"
#         "Return ONLY the number, no words, no punctuation.\n\n"
#         f"CANDIDATE_ANSWER:\n{candidate}\n\n"
#         "Number:"
#     )
#     inputs = _JTOK(prompt, return_tensors="pt").to(JUDGE_DEVICE)
#     with torch.inference_mode():
#         out = JUDGE.generate(
#             **inputs,
#             max_new_tokens=12,
#             do_sample=False,
#             temperature=0.0,
#         )
#     judge_text = _JTOK.decode(out[0], skip_special_tokens=True).strip()
#     return _last_int(judge_text)

def _to_text(x: Any) -> str:
    if isinstance(x, str): return x
    if isinstance(x, list) and x and isinstance(x[0], dict) and "content" in x[0]:
        return x[0]["content"]
    return str(x)

def rg_reward(completions: List[Any], ground_truth: List[dict], **_):
    rewards = []
    show_debug = (random.random() < 1)
    for c, gold in zip(completions, ground_truth):
        print("\n")
        print("-"*30)
        cand = _to_text(c)
        # print(cand)
        gold = str(gold).strip()
        print(f"actual answer:{gold}")
        # 1) Try strict extraction from candidate
        pred = extract_answer(c)
        print(f"model answer:{pred}")
        # # 2) If not found, ask judge (candidate only)
        # if pred is None:
        #     jnum = judge_compare(cand)
        #     pred = jnum
        #     print(f"judge pred: {pred}")
        # 3) Compare in Python
        try:
            correct = (pred is not None) and (pred == gold)
        except Exception:
            correct = False
        # assign that as the reward
        r = 1.0 if correct else 0.0

        # 4) Here is a custom change to the reward function
        # I want to try and use the SentenceBert approach from my dissertation to give the model a mild reward for getting close to the right answer
        # rather than a binary yes no - which for a small model like this could take forever before it begins to learn anything useful
        if pred is None:
            pred = 0
        if pred != 0:
            print("running custom reward process")
            # encode the correct answer
            # next you should encode the text using the embedding model
            correct_answer_embedding = sbert_model.encode(str(gold))
            # encode the model prediction
            model_answer_embedding = sbert_model.encode(str(pred))

            # get the similarity value
            search_results = sbert_model.similarity(correct_answer_embedding, model_answer_embedding)
            # return a dataframe with the semantic distance from the query to the location descriptions
            search_results_df = pd.DataFrame(search_results).T
            search_results_df.columns = ["reward_value"]
            sbert_r = round(search_results_df["reward_value"].values[0],4)

            # a simple alternative is to use levenshtein distance via fuzzy matching
            fuzz_r = (fuzz.ratio(str(gold).strip(),str(pred).strip() )/100)

            print(f"SBERT reward: {sbert_r}")
            print(f"FUZZ reward: {fuzz_r}")

            if r != 1:
                r = sbert_r
                r = fuzz_r
                r = min(sbert_r, fuzz_r)
                if r < 0.81:
                    # now exaggerate the reward signal for any values below 0.51
                    r = np.round(float(exaggerate_reward_signal(r)),2)

        print("reward:",r)
        rewards.append(float(r))

        if r == 1:
            print(Fore.GREEN + "Correct!")
            print(Style.RESET_ALL)
        elif r > 0:
            print(Fore.YELLOW + "Partially correct!")
            print(Style.RESET_ALL)
        else:
            print(Fore.RED + "Incorrect!")
            print(Style.RESET_ALL)

        if show_debug:
            # print(c)
            print("\n[LLM-JUDGE DEBUG]")
            print("GT:", gold)
            print("RAW:", cand[:280].replace("\n"," ") + ("..." if len(cand) > 280 else ""))
            print("PRED:", pred, "REWARD:", r)
            show_debug = False
    return rewards

# ----------------------------
# GRPO config (safe defaults)
# ----------------------------
args = GRPOConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_train_epochs=1,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=256,
    remove_unused_columns=False,
    fp16=False, bf16=False,         # FP32
    logging_steps=5,
    save_steps=200,
    report_to="none",
    # save_strategy="epoch",   # instead of "steps"
    # save_total_limit=3,      # keep only last 3 checkpoints
)

# Divisibility check
world_size = 1
try:
    from accelerate import Accelerator
    world_size = Accelerator().num_processes
except Exception:
    pass
eff = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
if eff % args.num_generations != 0:
    raise ValueError(f"Effective batch {eff} must be a multiple of num_generations {args.num_generations}")
print(f"[GRPO] world_size={world_size} | bs={args.per_device_train_batch_size} | "
      f"gas={args.gradient_accumulation_steps} | gens={args.num_generations} -> eff={eff}")

# ----------------------------
# Train
# ----------------------------
trainer = GRPOTrainer(
    model=policy,
    args=args,
    train_dataset=hf_ds,
    reward_funcs=rg_reward,
    processing_class=tok,
)

# if __name__ == "__main__":
#     trainer.train()
#     trainer.save_model(OUT_DIR)
#     print(f"Saved model to: {OUT_DIR}")

import time

if __name__ == "__main__":
    start_time = time.time()
    MAX_TRAIN_SECONDS = 2 * 10 * 60   # 2 hours

    trainer.train()

    elapsed = time.time() - start_time
    if elapsed > MAX_TRAIN_SECONDS:
        print(f"⏸️ Stopping early after {elapsed/60:.1f} minutes")
        # time.sleep(300)

    trainer.save_model(OUT_DIR)
    print(f"Saved model to: {OUT_DIR}")