from datasets import load_dataset
from transformers import  AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_LEN = 1024

# Create a tokenizer instance to use for filtering
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset and omit unused columns
dataset = load_dataset("hynky/czech_news_dataset_v2")
dataset = dataset["train"]
dataset = dataset.select_columns(["content", "brief"])


SYSTEM_PROMPT = (
    "Jsi profesionální český novinář a editor. Tvým úkolem je vytvořit výstižné, "
    "přesné a neutrální shrnutí dodaného novinového článku. Shrnutí musí obsahovat "
    "nejdůležitější fakta a nesmí si vymýšlet žádné nové informace."
)

def preprocess_batch(batch):
    prompts = []
    full_texts = []
    for content, brief in zip(batch["content"], batch["brief"]):

        if content is None or brief is None:
            prompts.append("")
            full_texts.append("")
            continue

        user_prompt = (
            "Přečti si následující článek a napiš jeho stručné shrnutí "
            "(maximálně 2 až 3 věty).\n\n"
            f"ČLÁNEK:\n{content}\n\n"
        )

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        full_text = prompt + brief + tokenizer.eos_token

        prompts.append(prompt)
        full_texts.append(full_text)

    tokens = tokenizer(
        full_texts,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )

    lengths = [len(ids) for ids in tokens["input_ids"]]

    return {
        "prompt": prompts,
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "length": lengths
    }


dataset = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=1000,
    num_proc=8,
)

dataset = dataset.filter(
    lambda x: x["length"] <= MAX_LEN,
    num_proc=8
)

# dataset = dataset.remove_columns(["length"])
# Smaller dataset version without the generated prompt and input_ids for testing 
dataset = dataset.remove_columns(["length", "attention_mask", "input_ids", "prompt"])



dataset.to_parquet("czech_news_sft.parquet")

dataset.shuffle(seed=42).select(range(100000)).to_parquet("czech_news_sft_100k.parquet")
dataset.shuffle(seed=42).select(range(50000)).to_parquet("czech_news_sft_50k.parquet")
dataset.shuffle(seed=42).select(range(10000)).to_parquet("czech_news_sft_10k.parquet")

# def filter_long_articles(x, max_length=2048):
#     # Max length of 2048 tokens (article + brief + prompt)
#     content = x.get("content")
#     brief = x.get("brief")

#     # Skip examples with missing fields
#     if content is None or brief is None:
#         return False
    
#     # Tokenize for length check
#     combined_text = x['content'] + " " + x['brief']
#     token_ids = tokenizer(combined_text, add_special_tokens=False)["input_ids"]
    
#     return len(token_ids) <= max_length

# # Tokenizing function to to get the full token length of the prompt containing the article and the brief
# def tokenize(x, tokenizer):
#     full_article = x.get("content") or ""
#     brief = x.get("brief") or ""

#     # Prompt definition
#     system_prompt = f"Jsi profesionální český novinář a editor. Tvým úkolem je vytvořit výstižné, přesné a neutrální shrnutí dodaného novinového článku. Shrnutí by mělo obsahovat nejdůležitější fakta a nesmí si vymýšlet žádné nové informace."
#     user_prompt = f"Přečti si následující článek a napiš jeho stručné shrnutí (maximálně 2 až 3 věty).\n\nČLÁNEK:\n{full_article}\n\n SHRNUTÍ:"

#     system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
#     user_text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
#     assistant_text = f"<|im_start|>assistant\n{brief}<|im_end|>"

#     # Tokenize the prompt components
#     system_tokens = tokenizer(system_text, add_special_tokens=False)
#     user_tokens = tokenizer(user_text, add_special_tokens=False)
#     assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)

#     input_ids = (
#         system_tokens["input_ids"]
#         + user_tokens["input_ids"]
#         + assistant_tokens["input_ids"]
#         + [tokenizer.eos_token_id]
#     )

#     return {"input_ids": input_ids}


# # Load the dataset
# dataset = load_dataset("hynky/czech_news_dataset_v2")

# # Create a tokenizer instance to use for filtering
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token



# # Select the necessary columns
# columns_to_keep = ["headline", "content", "brief"]
# truncated_dataset = dataset.select_columns(columns_to_keep)
# truncated_dataset = truncated_dataset.filter(
#     lambda x: x["content"] is not None and x["brief"] is not None
# )

# # Filter out articles that are too long
# truncated_dataset_tokenized = truncated_dataset.map(lambda x: tokenize(x, tokenizer))
# filtered_dataset = truncated_dataset_tokenized.filter(filter_long_articles)

# # Save full dataset with shorter articles and its subsets
# filtered_dataset.to_parquet("czech_news_filtered.parquet")
# filtered_dataset.shuffle(seed=42).select(range(100000)).to_parquet("czech_news_filtered_100k.parquet")
# filtered_dataset.shuffle(seed=42).select(range(50000)).to_parquet("czech_news_filtered_50k.parquet")
# filtered_dataset.shuffle(seed=42).select(range(10000)).to_parquet("czech_news_filtered_10k.parquet")

