import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

TABLE_EXTRACTION_PROMPT_HISTORY = [
    SystemMessage(
        "根据问题，提取表格中的内容，并使用 JSON 输出。`_thought`中的内容是你的思考过程，为特殊字段。在 JSON 中不要包含表格中无法找到对应答案的字段。"
    ),
    HumanMessage(
        '表格：<table><tr>三个苹果</tr><tr>四个香蕉</tr></table>\n\n\n问题：水果的总质量。回答格式：{{"mass": "..."}}\n'
    ),
    AIMessage(json.dumps({"_thought": "表格中没有提到质量，所以质量应该是未知。"})),
    HumanMessage(
        '表格：<table><tr>三个苹果</tr><tr>四个香蕉</tr></table>\n\n\n问题：请提供水果名称，总数量，总质量。回答格式：{{"name": ["..."], "total_amount": "...", "mass": "..."}}'
    ),
    AIMessage(
        json.dumps(
            {
                "_thought": "表格中提到了苹果和香蕉，所以水果名称应该包括苹果和香蕉。表格中提到了三个苹果和四个香蕉，所以总数量应该是7。表格中没有提到质量，所以质量应该是未知。",
                "name": ["苹果", "香蕉"],
                "total_amount": 7,
            }
        )
    ),
]

TABLE_EXTRACTION_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "表格：{table}\n\n\n问题：{question}"),
    ]
)
