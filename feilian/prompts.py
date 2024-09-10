import json
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


QUESTION_CONSTRUCTION_CN = (
    "根据问题的答案和参考资料，构造问题。问题中禁止透露参考资料中的任何实体、年份、地域等信息，问题需要简单且明确，并指定回答的格式（JSON）。得到该问题的受试者能够仅凭问题和类似资料，回答出符合格式要求的答案。\n"
    "\n"
    "答案：\n"
    "{answer_str}\n"
    "\n"
    "参考资料：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "你构造的问题是（直接提问，禁止回答与提问无关的内容）: "
)

QUESTION_CONSTRUCTION_EN = (
    "Construct a question based on the answer and the context. The question should be simple and clear and specify the format of the answer (JSON). A JSON should always be attached in the question with given key and doted values like `...`. The question should not mention any entity, date, time, company or location. The subject should be able to answer the question correctly with only the question and given context.\n"
    "\n"
    "Answer:\n"
    "{answer_str}\n"
    "\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Your question is (ask directly, do not answer, explain or ask irrelevant questions): "
)

EXTRACTION_PROMPT_HISTORY = [
    SystemMessage(
        "根据问题，从给定的上下文（网页内容）中提取信息，使用 JSON 输出。`_thought`字段是你的思考过程，需始终包含在回答的 json 中。除`_thought`外，禁止包含与提问无关的字段。给定的上下文可能不完整或不包含答案，需根据问题进行推断。输出时，JSON 中不应包含没有提及的字段。"
    ),
    # HumanMessage(
    #     '上下文：```<table><tr>三个苹果</tr><tr>四个香蕉</tr></table>````\n\n\n问：水果的总质量。回答格式：{{"mass": "..."}}\n'
    # ),
    # AIMessage(json.dumps({"_thought": "表格中没有提到质量，所以质量应该是未知。"})),
    # HumanMessage(
    #     '上下文：```<table><tr>三个苹果</tr><tr>四个香蕉</tr></table>```\n\n\n问题：请提供水果名称，总数量，总质量。回答格式：{{"name": ["..."], "total_amount": "...", "mass": "..."}}'
    # ),
    # AIMessage(
    #     json.dumps(
    #         {
    #             "_thought": "表格中提到了苹果和香蕉，所以水果名称应该包括苹果和香蕉。表格中提到了三个苹果和四个香蕉，所以总数量应该是7。表格中没有提到质量，所以质量应该是未知。",
    #             "name": ["苹果", "香蕉"],
    #             "total_amount": 7,
    #         }
    #     )
    # ),
]

EXTRACTION_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "上下文：```{context}```\n\n\n问题：{query}"),
    ]
)

XPATH_PROGRAM_PROMPT_HISTORY_CN = [
    SystemMessage(
        "根据问题，编写提取各个字段的 XPath 规则，并使用 JSON 输出。`_thought`字段是你的思考过程，需始终包含在回答的 json 中。"
    ),
    HumanMessage(
        '```html\n<table><tr><td>3个</td><td>苹果</td></tr><tr><tr><td>4个</td><td>香蕉</td></tr></table>\n```\n\n\n问题：几个柠檬？\n回答格式：{{"n_lemons": "..."}}'
    ),
    AIMessage(
        json.dumps(
            {
                "_thought": "表格中提到了苹果和香蕉，苹果数量是10，香蕉数量是3。编写 XPath 时，应该注意区分不同的水果，并使用 regex，以便提取正确的数量且有鲁棒性。",
                "n_apples": "//td[re:test(text(), '苹果')]/../td[1]/text()",
                "n_bananas": "//td[re:test(text(), '香蕉')]/../td[1]/text()",
            }
        )
    ),
]

XPATH_PROGRAM_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "```html\n{html}\n```\n\n\n问题：{query}"),
    ]
)

QUESTION_CONVERSION_COMP_CN = PromptTemplate.from_template(
    (
        "你需要将用户提问转换为编写 XPath 的任务描述，并保留格式要求。新的格式需要使键与原格式要求完全一致，值始终为字符串，并添加`_thought`字段，并使其位于其他键之前。仅转换问题，不要回答。"
        "\n\n"
        "示例：\n"
        "问题：请提供一个商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{{'name':['商学院名称'],'phone':['联系电话'],'type':['类型'],'website':['网站链接']\n}}```"
        "转换后：编写 XPath 用以提取商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{{'_thought':'...','name':'...','phone':'...','type':'...','website':'...'\n}}"
        "\n\n"
        "用户的问题：{query}"
        "准换后："
    )
)

# 对话形式的问题任务效果不行，补全形式的更好
QUESTION_CONVERSION_CN = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "你需要将用户提问转换为编写 XPath 的任务描述，并保留格式要求。新的格式需要使键与原格式要求完全一致，值始终为字符串，并添加`_thought`字段，并使其位于其他键之前。仅转换问题，不要回答。"
        ),
        HumanMessage(
            '请提供一个商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{"name":["商学院名称"],"phone":["联系电话"],"type":["类型"],"website":["网站链接"]\n}```'
        ),
        AIMessage(
            '编写 XPath 用以提取商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{"_thought":"...","name":"...","phone":"...","type":"...","website":"..."\n}'
        ),
        HumanMessage("{query}"),
    ]
)
