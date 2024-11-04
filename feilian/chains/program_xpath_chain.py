import os
import json_repair
from functional import compose
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic

from feilian.prompts import (
    XPATH_PROGRAM_PROMPT_HISTORY_CN,
    XPATH_PROGRAM_PROMPT_CN,
    COT_PROGRAM_XPATH_PROMPT_CN,
)


def _create_program_xpath_chain():
    def parser(response):
        return json_repair.repair_json(
            response.content,
            return_objects=True,
        )

    llm = ChatOpenAI( #ChatAnthropic
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return (
        {
            "query": itemgetter("query"),
            "html0": compose(itemgetter(0), itemgetter("htmls")),
            "html1": compose(itemgetter(1), itemgetter("htmls")),
            "html2": compose(itemgetter(2), itemgetter("htmls")),
        }
        | XPATH_PROGRAM_PROMPT_CN.partial(chat_history=XPATH_PROGRAM_PROMPT_HISTORY_CN)
        | llm
        | parser
    )


def _create_cot_program_xpath_s3_chain():
    def parser(response):
        json_str = response.content.split("最终结论:")[-1].strip()
        json_str = json_str.split("```")[0].strip()
        return json_repair.repair_json(json_str, return_objects=True)

    llm = ChatOpenAI( # ChatAnthropic
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return (
        {
            "query": itemgetter("query"),
            "html0": compose(itemgetter(0), itemgetter("htmls")),
            "html1": compose(itemgetter(1), itemgetter("htmls")),
            "html2": compose(itemgetter(2), itemgetter("htmls")),
        }
        | PromptTemplate.from_template(COT_PROGRAM_XPATH_PROMPT_CN)
        | llm
        | parser
    )


def create_cot_program_xpath_s2():
    lang = os.environ.get("PROMPT_LANG", "en")

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

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0) # ChatAnthropic
    template_file = f"feilian/prompts/{lang}/program_xpath_cot_s2.yaml"
    return (
        {
            "html0": compose(itemgetter(0), itemgetter("htmls")),
            "html1": compose(itemgetter(1), itemgetter("htmls")),
            "data0": compose(itemgetter(0), itemgetter("datas")),
            "data1": compose(itemgetter(1), itemgetter("datas")),
        }
        | PromptTemplate.from_file(template_file)
        | llm
        | parser
    )


def create_cot_program_xpath_s1():
    def parser(response):
        json_str = response.content.split("最终结论:")[-1].strip()
        if json_str.startswith("```json"):
            json_str = json_str.split("```json")[1].strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1].strip()
        json_str = json_str.split("```")[0].strip()
        return json_repair.repair_json(json_str, return_objects=True)

    llm = ChatOpenAI( # ChatAnthropic
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return (
        {
            "html": compose(itemgetter(0), itemgetter("htmls")),
            "data": compose(itemgetter(0), itemgetter("datas")),
        }
        | PromptTemplate.from_file("feilian/prompts/cn/program_xpath_cot_s1.yaml")
        | llm
        | parser
    )


program_xpath = _create_program_xpath_chain()
cot_program_xpath = _create_cot_program_xpath_s3_chain()
cot_program_xpath_s2 = create_cot_program_xpath_s2()
cot_program_xpath_s1 = create_cot_program_xpath_s1()
