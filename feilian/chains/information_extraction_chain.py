import os
import re
import json_repair
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def create_information_extraction_chain():
    lang = os.environ.get("PROMPT_LANG", "cn")

    def parser(response):
        if lang == "cn":
            json_str = response.content.split("最终结论:")[-1].strip()
        else:
            json_str = response.content.split("Final Conclusion:")[-1].strip()

        if json_str.startswith("```json"):
            json_str = json_str.split("```json")[1].strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1].strip()
        json_str = json_str.split("```")[0].strip()
        return json_repair.repair_json(json_str, return_objects=True)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
    template_file = f"feilian/prompts/{lang}/information_extraction.yaml"
    return PromptTemplate.from_file(template_file) | llm | parser


def _create_best_composition_chain():
    lang = os.environ.get("PROMPT_LANG", "cn")

    def parser(response):
        if lang == "cn":
            choices = response.content.split("最终结论:")[-1].strip()
        else:
            choices = response.content.split("Final Conclusion:")[-1].strip()

        choices = choices.split("\n")[0]
        choices = [int(x) for x in re.findall(r"\d+", choices)]

        return choices

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
    template_file = f"feilian/prompts/{lang}/best_composition.yaml"
    prompt = PromptTemplate.from_file(template_file)
    return prompt | llm | parser


best_composition_chain = _create_best_composition_chain()
information_extraction_chain = create_information_extraction_chain()
