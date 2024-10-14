import os
import json_repair
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

lang = os.environ.get("PROMPT_LANG", "cn")

program_xpath_chat_system = PromptTemplate.from_file(
    "feilian/prompts/cn/program_xpath_chat_system.jinja2",
    template_format="jinja2",
)

program_xpath_chat_user = PromptTemplate.from_file(
    "feilian/prompts/cn/program_xpath_chat_user.jinja2",
    template_format="jinja2",
)

program_xpath_chat_feedback = PromptTemplate.from_file(
    "feilian/prompts/cn/program_xpath_chat_feedback.jinja2",
    template_format="jinja2",
)


def format_snippets(snippets):
    return program_xpath_chat_user.format(snippets=snippets)


def format_feedbacks(feedbacks):
    return program_xpath_chat_feedback.format(feedbacks=feedbacks)


store = {}
cached_session_ids = []


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    if session_id not in cached_session_ids:
        cached_session_ids.append(session_id)
        if len(cached_session_ids) > 128:
            del store[cached_session_ids.pop(0)]

    return store[session_id]


def create_program_xpath_chat_chain():
    def parser(response):
        text = response.content
        print("AI: ", text)

        if text.startswith("```json"):
            text = text.split("```json")[1].strip()
        if text.endswith("```"):
            text = text.split("```")[0].strip()

        if text.startswith("/"):
            return text

        try:
            result = json_repair.loads(text) or {}
        except Exception as e:
            print(f"Error: {e}")
            return ""

        if isinstance(result, str):
            return result
        elif isinstance(result, list) and result and isinstance(result[0], str):
            return result[0]
        return result.get("xpath", "")

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        max_tokens=512,
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", program_xpath_chat_system.template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{{ input }}"),
        ],
        template_format="jinja2",
    )

    return RunnableWithMessageHistory(
        chat_template | llm | parser,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


program_xpath_chat_chain = create_program_xpath_chat_chain()
