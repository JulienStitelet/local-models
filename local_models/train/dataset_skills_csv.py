import os
import json
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file
from langchain_pydantic_minifier.minifier_pydantic import MinifiedPydanticOutputParser

from local_models.prompts import get_prompt
from local_models.schema.resume import SkillsResponse

RESUMES = "./resources/raw_resumes"
EXPECTED = "./resources/openai_skills"
DATASET = "./resources/dataset"

## ENV ===> HUGGINGFACE_TOKEN="hf_wwwwwww"

conversations = []
parser = MinifiedPydanticOutputParser(pydantic_object=SkillsResponse)

for filename in os.listdir(EXPECTED):
    filepath = os.path.join(EXPECTED, filename)
    if os.path.isfile(filepath):
        # Do your thing here
        print(f"Processing {filepath}")
        file_id = os.path.splitext(os.path.basename(filepath))[0]
        with open(f"{RESUMES}/{file_id}.txt", "r", encoding="utf-8") as raw_resume:
            with open(f"{EXPECTED}/{file_id}.json", "r", encoding="utf-8") as expected:
                prompt = get_prompt(parser, raw_resume.read())
                json_dict = json.load(expected)
                expected_output = f"```json{json.dumps(json_dict)}```"
                conversations.append(
                    {
                        "text": f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{expected_output}<end_of_turn>"
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
