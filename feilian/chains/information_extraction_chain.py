import os
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


information_extraction_chain = create_information_extraction_chain()
