import os
import json_repair
from functional import compose
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def create_cot_program_css_selector_s2():
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
        {
            "html0": compose(itemgetter(0), itemgetter("htmls")),
            "html1": compose(itemgetter(1), itemgetter("htmls")),
            "data0": compose(itemgetter(0), itemgetter("datas")),
            "data1": compose(itemgetter(1), itemgetter("datas")),
        }
        | PromptTemplate.from_file("feilian/prompts/cn/program_selector_cot_s2.yaml")
        | llm
        | parser
    )


def create_cot_program_css_selector_s1():
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
        {
            "html": compose(itemgetter(0), itemgetter("htmls")),
            "data": compose(itemgetter(0), itemgetter("datas")),
        }
        | PromptTemplate.from_file("feilian/prompts/cn/program_selector_cot_s1.yaml")
        | llm
        | parser
    )


cot_program_css_selector_s2 = create_cot_program_css_selector_s2()
cot_program_css_selector_s1 = create_cot_program_css_selector_s1()
