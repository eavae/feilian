import os
import re
import json_repair
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

lang = os.environ.get("PROMPT_LANG", "cn")


def json_parser(response):
    if lang == "cn":
        json_str = response.content.split("结论:")[-1].strip()
    else:
        json_str = response.content.split("Conclusion:")[-1].strip()

    if json_str.startswith("```json"):
        json_str = json_str.split("```json")[1].strip()
    if json_str.startswith("```"):
        json_str = json_str.split("```")[1].strip()
    json_str = json_str.split("```")[0].strip()
    json_str = json_str or "{}"
    return json_repair.repair_json(json_str, return_objects=True)


def create_information_extraction_chain(cue=False):
    llm = ChatOpenAI(model=os.getenv("OPENAI_IE_MODEL"), temperature=0)
    if cue:
        template_file = f"feilian/prompts/{lang}/information_extraction_with_cue.jinja2"
        example = {
            "title": {"value": ["extracted title"], "cue_text": ""},
            "phone": {"value": ["1234567890"], "cue_text": "电话号码"},
        }
    else:
        template_file = f"feilian/prompts/{lang}/information_extraction.jinja2"
        example = {
            "title": ["extracted title 1", "extracted title 2"],
            "phone": ["1234567890"],
        }

    return (
        {
            "query": itemgetter("query"),
            "context": itemgetter("context"),
            "json_example": lambda _: example,
        }
        | PromptTemplate.from_file(template_file, template_format="jinja2")
        | llm
        | json_parser
    )


def _create_best_composition_chain():
    lang = os.environ.get("PROMPT_LANG", "cn")

    def parser(response):
        if lang == "cn":
            choices = response.content.split("最终结论:")[-1].strip()
        else:
            choices = response.content.split("Final Conclusion:")[-1].strip()

        choices = choices.split("\n")[0]
        choices = [int(x) for x in re.findall(r"\d", choices)]

        return choices

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
    template_file = f"feilian/prompts/{lang}/best_composition.yaml"
    prompt = PromptTemplate.from_file(template_file)
    return prompt | llm | parser


best_composition_chain = _create_best_composition_chain()
information_extraction_chain = create_information_extraction_chain()
cued_information_extraction_chain = create_information_extraction_chain(cue=True)
