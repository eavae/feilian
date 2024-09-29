import os
import json_repair
from functional import compose
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def _create_critic_xpath_chain():
    def parser(response):
        return json_repair.repair_json(
            response.content,
            return_objects=True,
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return (
        {
            "context": itemgetter("context"),
            "xpath": itemgetter("xpath"),
            "ground_truth1": compose(itemgetter(0), itemgetter("ground_truths")),
            "ground_truth2": compose(itemgetter(1), itemgetter("ground_truths")),
            "ground_truth3": compose(itemgetter(2), itemgetter("ground_truths")),
            "prediction1": compose(itemgetter(0), itemgetter("predictions")),
            "prediction2": compose(itemgetter(1), itemgetter("predictions")),
            "prediction3": compose(itemgetter(2), itemgetter("predictions")),
        }
        | PromptTemplate.from_file("feilian/prompts/critic_xpath.yaml")
        | llm
        | parser
    )


critic_xpath_chain = _create_critic_xpath_chain()
critic_xpath_chain_en = _create_critic_xpath_chain()
