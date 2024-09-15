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
        "根据问题，编写提取各个字段的 XPath，并使用 JSON 输出。当有多个 html 片段时，你需要针对某个字段编写**一个xpath**以适应所有具有相似结构的 html 片段。`_thought`字段是你的思考过程，需始终包含在回答的 json 中。你需要逐字段思考，并根据用户提供的 html 片段，推理你的思考过程。你编写的 xpath 应简洁明了，以适应具有相似结构的网页。注意，与标准 XPath 不同，这里可使用 regex，比如`re:test` 或 `re:match`来增加xpath的泛用性。编写 xpath 时，优先使用结构及标签语义。使用regex时，仅保留**1-2个关键单词**，以便在相似结构但内容不同的网站上使用。"
    ),
    HumanMessage(
        '```html\n<table><tr><td>3个</td><td>苹果</td></tr><tr><tr><td>4个</td><td>香蕉</td></tr></table>\n```\n\n\n问题：几个香蕉、苹果？\n回答格式：{{"n_apples": "...", "n_bananas": "..."}}'
    ),
    AIMessage(
        json.dumps(
            {
                "_thought": "首先，表格中提到了苹果和香蕉，苹果数量是10，香蕉数量是3。接下来，我需要编写两个 XPath 用以提取苹果和香蕉的数量。1. 苹果，苹果出现在 td 中，表明该元素在表格中，且父级有一个  tr 标签。先获取表格的一行，然后寻找第二个 td 标签会更符合html的语义逻辑，所以我可以使用 `//td[re:test(text(), '苹果')]/../td[2]/text()` 来提取苹果的数量。2. 香蕉，同上，我可以使用 `//td[re:test(text(), '香蕉')]/../td[2]/text()` 来提取香蕉的数量。",
                "n_apples": "//td[re:test(text(), '苹果')]/../td[2]/text()",
                "n_bananas": "//td[re:test(text(), '香蕉')]/../td[2]/text()",
            }
        )
    ),
]

XPATH_PROGRAM_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "```html\n{html0}\n```\n\n```html\n{html2}\n```\n\n```html\n{html2}\n\n\n问题：{query}",
        ),
    ]
)

QUESTION_CONVERSION_COMP_CN = PromptTemplate.from_template(
    (
        "你需要将用户提问转换为编写 XPath 的任务描述，并保留格式要求。新的格式需要使键与原格式要求完全一致，值始终为字符串（禁止使用数组），并添加`_thought`字段，并使其位于其他键之前。仅转换问题，不要回答。"
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
