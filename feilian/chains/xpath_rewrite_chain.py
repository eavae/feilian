import os
import json_repair
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic

lang = os.environ.get("PROMPT_LANG", "en")


def rindex(lst, value):
    return len(lst) - lst[::-1].index(value[::-1])


def parse_with_conclusion(response):
    if lang == "cn":
        idx = rindex(response.content, "最终结论:")
    else:
        idx = rindex(response.content, "Final Conclusion:")
    xpath = response.content[idx:].strip()

    if xpath.startswith("`"):
        xpath = xpath.split("`")[1].strip()
    if xpath.endswith("`"):
        xpath = xpath.split("`")[0].strip()

    print(response.content)
    return xpath


def create_xpath_rewrite_chain():
    def parser(response):
        text = response.content
        print(text)
        if text.startswith("```json"):
            text = text.split("```json")[1].strip()
        if text.endswith("```"):
            text = text.split("```")[0].strip()

        result = json_repair.loads(text) or {}
        return result.get("xpath", "")

    llm = ChatOpenAI( # ChatAnthropic
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return (
        PromptTemplate.from_file(
            "feilian/prompts/cn/xpath_rewrite.yaml",
            template_format="jinja2",
        )
        | llm
        | parser
    )


rewrite_xpath_chain = create_xpath_rewrite_chain()
