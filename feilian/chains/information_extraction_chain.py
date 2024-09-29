import os
import re
import json_repair
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def create_information_extraction_chain():
    def parser(response):
        json_str = response.content.split("最终结论:")[-1].strip()
        if json_str.startswith("```json"):
            json_str = json_str.split("```json")[1].strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1].strip()
        json_str = json_str.split("```")[0].strip()
        return json_repair.repair_json(json_str, return_objects=True)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return (
        PromptTemplate.from_file("feilian/prompts/information_extraction.yaml")
        | llm
        | parser
    )


def _create_best_composition_chain():
    def parser(response):
        choices = response.content.split("最终结论:")[-1].strip()
        choices = choices.split("\n")[0]
        choices = [int(x) for x in re.findall(r"\d+", choices)]

        return choices

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
    prompt = PromptTemplate.from_file("feilian/prompts/best_composition.yaml")
    return prompt | llm | parser


best_composition_chain = _create_best_composition_chain()
information_extraction_chain = create_information_extraction_chain()
