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
