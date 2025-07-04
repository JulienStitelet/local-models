from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_pydantic_minifier.minifier_pydantic import MinifiedPydanticOutputParser
from local_models.schema.resume import SkillsResponse

DIRECTORY = "./resources/raw_resumes"
DIRECTORY_OUTPUT = "./resources/openai_skills"

model = ChatOpenAI(
    model="gpt-4o",
    request_timeout=30,
    temperature=0.0,
    max_retries=0,
)

EXTRACT_RESUME_PROMPT = """
Extract entities from this Resume sent in the query. 
Find which entities to extract from the structured output class injected with this query.
Use the 'description' in each field to know what to extract.
If you can't find the value for the field, leave it empty, never use null.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=EXTRACT_RESUME_PROMPT),
        MessagesPlaceholder(variable_name="history_list"),
        MessagesPlaceholder(variable_name="human_message", optional=True),
    ]
)

parser = MinifiedPydanticOutputParser(pydantic_object=SkillsResponse, strict=True)

chain = chat_prompt | model.with_structured_output(parser.minified, strict=True)


def get_openai_output(filepath: str) -> SkillsResponse:
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    print(f"processing {file_id} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_resume = f.read()
        placeholders = {"history_list": [], "human_message": [raw_resume]}
        llm_output = chain.invoke(placeholders)
        skills = parser.get_original(llm_output)
        with open(f"{DIRECTORY_OUTPUT}/{file_id}.json", "w", encoding="utf-8") as w:
            w.write(skills.model_dump_json())


def parallel_get_openai_outputs(
    resumes: list[str], max_workers: int = 4
) -> list[SkillsResponse]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_resume = {
            executor.submit(get_openai_output, resume): resume for resume in resumes
        }
        for future in as_completed(future_to_resume):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing resume: {e}")
    return results


files = sorted(os.listdir(DIRECTORY))
filepaths = []
for filename in files:
    filepath = os.path.join(DIRECTORY, filename)
    if os.path.isfile(filepath):
        filepaths.append(filepath)

parallel_get_openai_outputs(filepaths, max_workers=1)
