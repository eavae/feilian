# 该文件已废弃

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
]

EXTRACTION_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "上下文：```{context}```\n\n\n问题：{query}"),
    ]
)

BEST_ANSWERS_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "根据问题及给定的上下文（网页内容），选择一组能够恰好构成最佳答案的组合（可以**多选**）。提供答案时，仅列出序号即可，并用`,`分割，禁止输出除选项外的其他内容。"
        ),
        HumanMessage(
            '问题：```text\n有几个苹果和香蕉呢？\n```\n\n\n上下文：```text\n这里有3个苹果、4个香蕉。\n```\n\n\n待选答案：\n1. {{"n_apples": 3}}\n2. {{"n_bananas": 4}}\n3. {{}}'
        ),
        AIMessage("1, 2"),
        (
            "human",
            "问题：```text\n{query}\n```\n\n\n上下文：```text\n{context}\n```\n\n\n待选答案：\n{choices}",
        ),
    ]
)

BEST_COMPOSITION_CN = (
    "根据问题、给定的上下文（网页内容）和待选列表，选择一组能够恰好回答问题的最佳组合（可以**多选**）。注意，部分正确且无法被其它选项覆盖的选项，应被视为正确选项。\n"
    "\n"
    "回答时，请遵循如下格式：\n"
    "```text\n"
    "选项1: ...(你对选项1的分析及思考)...\n"
    "选项2: ...(你对选项2的分析及思考)...\n"
    "选项n: ...(你对选项n的分析及思考)...\n"
    "总结: ...(根据提问，判断提问者意图，并推断最合理的选项组合)...\n"
    "最终结论: ...(形成最终结论，仅列出序号即可，并用`,`分割)...\n"
    "```\n"
    "\n"
    "参考资料：\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "\n"
    "问题：\n"
    "{query}\n"
    "\n"
    "待选回答：\n"
    "{choices}\n"
    "\n"
    "你的分析过程、总结及结论:\n "
)

XPATH_PROGRAM_PROMPT_HISTORY_CN = [
    SystemMessage(
        '根据问题，编写提取各个字段的 XPath，并使用 JSON 输出。当有多个 html 片段时，你需要针对某个字段编写**一个xpath**以适应所有具有相似结构的 html 片段。`_thought`字段是你的思考过程，需始终包含在回答的 json 中。你需要逐字段思考，并根据用户提供的 html 片段，显示的推理并反思。注意点：1. 简洁优于复杂，优先使用 html 自身的语义规则，比如tag、class、id。2. `contains`优于精准匹配，比如`//div[@class="title"]`可写成`//div[contains(@class, "title")]`。3. 在xpath 中，必须使用**双引号**，并合理的转义单双引号。4. 适当使用 regex 进行复杂文本的匹配，比如`re:test`。5. `text()` 只能够获取**当前节点的文本**，而不是该节点及以下节点的所有文本，使用时注意该限制条件。'
    ),
    HumanMessage(
        '```html\n<table><tr><td>3个</td><td>苹果</td></tr><tr><tr><td>4个</td><td>香蕉</td></tr></table>\n```\n\n\n问题：几个香蕉、苹果？\n回答格式：{{"n_apples": "...", "n_bananas": "..."}}'
    ),
    AIMessage(
        json.dumps(
            {
                "_thought": "首先，表格中提到了苹果和香蕉，苹果数量是10，香蕉数量是3。接下来，我需要编写两个 XPath 用以提取苹果和香蕉的数量。1. 苹果，苹果出现在 td 中，表明该元素在表格中，且父级有一个  tr 标签。先获取表格的一行，然后寻找第二个 td 标签会更符合html的语义逻辑。2. 香蕉，同上，只需要将匹配的文本换成香蕉即可",
                "n_apples": '//td[contains(text(), "苹果")]/../td[2]/text()',
                "n_bananas": '//td[contains(text(), "香蕉")]/../td[2]/text()',
            }
        )
    ),
]

XPATH_PROGRAM_PROMPT_CN = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "```html\n{html0}\n```\n\n```html\n{html1}\n```\n\n```html\n{html2}\n\n\n问题：{query}",
        ),
    ]
)


COT_PROGRAM_XPATH_PROMPT_CN = (
    "根据问题，编写用于提取各个字段文本内容的 XPath。"
    "当有多个 html 片段时，你需要针对某个字段编写**一条xpath**以适应所有具有相似结构的 html 片段。\n"
    "\n"
    "XPATH最佳实践:\n"
    '1. 基于强属性（比如：id、class、name等）和标签自身的语义选择合适的**锚点**，比如：`//div[@id="example"]。`\n'
    '2. 选择恰当的函数进行辅助。比如contains()、starts-with()、text()、re:test()等，比如：`//*[re:test(., "^abc$", "i")]"`。\n'
    "3. 基于**锚点**，使用轴（axes）细致的游走到目标节点（无法直接通过1或2找到目标节点时），比如：`/button[@id='goodID']/parent::*/following-sibling::*[1]/button`。\n"
    "4. 使用`text()`函数时，注意它只能匹配**当前节点**（不包含子节点！）的文本，禁止使用长文本匹配。使用关键词和`contains`或`re:test`函数事半功倍。\n"
    '5. `contains`优于`=`，比如`//div[@class="title"]`可写成`//div[contains(@class, "title")]`\n'
    '6. `contains`的一个参数必须是某属性或函数，禁止类似`//div[contains(., "Some Text")]`的写法，推荐`//div[contains(text(), "Some Text")]`。\n'
    "\n"
    "回答时，严格遵循如下格式：\n"
    "```text\n"
    "结论[字段A]: ...(用于提取字段A的XPath)...\n"
    "批判[字段A]: ...(以批判式的思维方式，寻找结论中 XPath 的潜在问题或漏洞，必须参考最佳实践。以`/`或`//`拆分XPath，逐段进行批判并记录。根据批判的结果，得出是否合理的结论。若不合理，则对字段A进行下一轮`结论-批判`链，若合理，则分析下一个字段)...\n"
    "...（若字段A的结论不合理，重新对字段A得出结论并批判）..."
    "结论[字段A]: ...(对字段A重新得出的结论)...\n"
    "批判[字段A]: ...(对字段A重新批判)...\n"
    "...（若字段A的结论合理，继续分析下一个字段）..."
    "结论[字段B]: ...(对字段B得出的结论)...\n"
    "批判[字段B]: ...(对字段B结论的批判)...\n"
    "...（若所有字段都分析完毕，得出最终结论）...\n"
    '最终结论: ...(使用 JSON 格式回答，比如：{{"字段A":"//div[contains(@class, "exampleA")]","字段B":"//div[contains(@class, "exampleB")]"}})...\n'
    "```\n"
    "\n"
    "HTML片段：\n"
    "----------片段一----------\n"
    "{html0}\n\n\n"
    "----------片段二----------\n"
    "{html1}\n\n\n"
    "----------片段三----------\n"
    "{html2}\n\n\n"
    "\n"
    "问题：\n"
    "{query}\n"
    "\n\n\n"
    "逐字段进行 结论-反思 过程，并记录如下:\n "
)

BACKUPS = (
    "推理[字段A]: ...(观察HTML，记录现象，思考如何编写XPath用以提取字段A，并记录过程)...\n"
    "推理[字段A]: ...(若上一次批判结果不合理时，继续推理字段A，以修正问题)...\n"
    "推理[字段B]: ...(你对字段B的推理)...\n"
)

QUESTION_CONVERSION_COMP_CN = PromptTemplate.from_template(
    (
        "你需要将用户提问转换为编写 XPath 的任务描述，并保留格式要求。新的格式需要使键与原格式要求完全一致，值始终为字符串（禁止使用数组）。仅转换问题，不要回答。"
        "\n\n"
        "示例：\n"
        "问题：请提供一个商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{{'name':['商学院名称'],'phone':['联系电话'],'type':['类型'],'website':['网站链接']\n}}```"
        "转换后：编写 XPath 用以提取商学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{{name':'...','phone':'...','type':'...','website':'...'\n}}"
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
            '请提供一个学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{"name":["商学院名称"],"phone":["联系电话"],"type":["类型"],"website":["网站链接"]\n}```'
        ),
        AIMessage(
            '编写 XPath 用以提取学院的名称、联系电话、类型和网站链接，以JSON格式回答。\n\n```json\n{"_thought":"...","name":"...","phone":"...","type":"...","website":"..."\n}'
        ),
        HumanMessage("{query}"),
    ]
)
