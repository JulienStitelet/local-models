import os
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file

TESTS = "./resources/openai_skills"
INPUT = "./resources/raw_resumes"
DATASET = "./resources/dataset"

## ENV ===> HUGGINGFACE_TOKEN="hf_wwwwwww"

conversations = []

for filename in os.listdir(TESTS):
    filepath = os.path.join(TESTS, filename)
    if os.path.isfile(filepath):
        # Do your thing here
        print(f"Processing {filepath}")
        file_id = os.path.splitext(os.path.basename(filepath))[0]
        with open(f"{INPUT}/{file_id}.txt", "r", encoding="utf-8") as rr:
            with open(f"{TESTS}/{file_id}.json", "r", encoding="utf-8") as resp:
                conversations.append(
                    {
                        "text": f"<bos><start_of_turn>user\n{rr.readlines()}<end_of_turn>\n<start_of_turn>model\n{resp.readlines()}<end_of_turn>"
                    }
                )

df = pd.DataFrame(conversations)
df.to_parquet(f"{DATASET}/skills.parquet", engine="pyarrow", index=False)


repo_id = "julienstitelet/skills-dataset"

token = os.getenv("HF_TOKEN", "miaou")
# create_repo(repo_id, repo_type="dataset", private=True, token=token)

upload_file(
    path_or_fileobj=f"{DATASET}/skills.parquet",
    path_in_repo="skills.parquet",
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
)
# use it like:
# from datasets import load_dataset
# dataset = load_dataset("parquet", data_files="skills.parquet")
