import os
import json

DIRECTORY = "./resources/resumes"
RAW_SAVE_DIR = "./resources/raw_resumes"
RAW_NUMBER = 0


def save_raw_resume(raw_resume: str, file_id: int) -> None:
    with open(f"{RAW_SAVE_DIR}/{file_id}.txt", "w", encoding="utf-8") as f:
        f.write(raw_resume)


def extract_raw_resume(filepath, file_id: int) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if "successResponse" in data:
                success_response = json.loads(data["successResponse"])
                raw_resume = success_response["document"][0]["document"]
                print(raw_resume)
                save_raw_resume(raw_resume, file_id)
            else:
                print("bypass this file as no document inside")
        except Exception as e:
            print(f"error: {e}")


for filename in os.listdir(DIRECTORY):
    filepath = os.path.join(DIRECTORY, filename)
    if os.path.isfile(filepath):
        # Do your thing here
        print(f"Processing {filepath}")
        extract_raw_resume(filepath, RAW_NUMBER)
        RAW_NUMBER += 1
