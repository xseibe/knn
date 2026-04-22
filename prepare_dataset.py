from datasets import load_dataset
from transformers import  AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_LEN = 1024

# Create a tokenizer instance to use for filtering
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset and omit unused columns
dataset = load_dataset("hynky/czech_news_dataset_v2")
dataset = dataset["train"]
dataset = dataset.select_columns(["content", "brief"])

# Filter out examples with missing articles or briefs
dataset = dataset.filter(
    lambda x: x["content"] is not None and x["brief"] is not None
              and x["content"].strip() != ""
              and x["brief"].strip() != "",
    num_proc=8
)


SYSTEM_PROMPT = (
    "Jsi profesionální český novinář a editor. Tvým úkolem je vytvořit výstižné, "
    "přesné a neutrální shrnutí dodaného novinového článku. Shrnutí musí obsahovat "
    "nejdůležitější fakta a nesmí si vymýšlet žádné nové informace."
)

# Generate training-style prompt and get its token length for filtering
def preprocess_batch(batch):
    lengths = []
    for content, brief in zip(batch["content"], batch["brief"]):

        if content is None or brief is None:
            lengths.append(0)
            continue

        system_prompt = f"Jsi profesionální český novinář a editor. Tvým úkolem je vytvořit výstižné, přesné a neutrální shrnutí dodaného novinového článku. Shrnutí by mělo obsahovat nejdůležitější fakta a nesmí si vymýšlet žádné nové informace."
        user_prompt = f"Přečti si následující článek a napiš jeho stručné shrnutí (maximálně 2 až 3 věty).\n\nČLÁNEK:\n{content}:"

        system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        user_text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        assistant_prefix = "<|im_start|>assistant\n"
        assistant_suffix = "<|im_end|>"

        # Tokenize the prompt components
        system_tokens = tokenizer(system_text, add_special_tokens=False)
        user_tokens = tokenizer(user_text, add_special_tokens=False)
        assistant_prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)
        assistant_tokens = tokenizer(brief, add_special_tokens=False)
        assistant_suffix_tokens = tokenizer(assistant_suffix, add_special_tokens=False)

        input_ids = (
            system_tokens["input_ids"]
            + user_tokens["input_ids"]
            + assistant_prefix_tokens["input_ids"]
            + assistant_tokens["input_ids"]
            + assistant_suffix_tokens["input_ids"]
            + [tokenizer.eos_token_id]
        )

        lengths.append(len(input_ids))

    return {"length": lengths}


# Map the preprocessing function to the dataset
dataset = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=1000,
    num_proc=8,
)

# Filter out longer examples 
dataset = dataset.filter(
    lambda x: x["length"] <= MAX_LEN,
    num_proc=8
)
dataset = dataset.remove_columns(["length"])

# Smaller dataset version without the generated prompt and input_ids for testing 
dataset.to_parquet("czech_news_sft.parquet")
dataset.shuffle(seed=42).select(range(100000)).to_parquet("czech_news_sft_100k.parquet")
dataset.shuffle(seed=42).select(range(50000)).to_parquet("czech_news_sft_50k.parquet")
dataset.shuffle(seed=42).select(range(10000)).to_parquet("czech_news_sft_10k.parquet")