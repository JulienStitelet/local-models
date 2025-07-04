from langchain_core.prompts import PromptTemplate


def get_prompt(parser, raw_resume):
    chat_prompt = PromptTemplate(
        template="{system_prompt}\n\n{format_instructions}\n\nHere is the resume:\n{query}\n.## IMPORTANT:{important}",
        input_variables=["human_message"],
        partial_variables={
            "system_prompt": "Extract information from the given raw text resume. Wrap the output in `json` tags",
            "format_instructions": parser.get_format_instructions(),
            "important": """Return a JSON object containing only fields that have meaningful values.

    - Do NOT include fields with null, empty strings, empty arrays, or missing data.
    - Omit any field for which no reliable value is found.
    - For example, if the value for "d" is not available, do not include "d" at all in the JSON output.

    The final JSON should be as compact as possible with only non-empty, non-null fields.
    Field names must be lower case""",
        },
    )

    prompt = chat_prompt.invoke({"query": raw_resume}).to_string()
    print(prompt)
    return prompt
