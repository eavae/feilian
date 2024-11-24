def is_openai_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("gpt-") or model_name.startswith("o1-")


def is_deepseek_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepseek-")


def is_anthropic_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("claude-")


def is_google_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("gemini-")


def check_model(model_name):
    if is_openai_model(model_name):
        return "openai"
    if is_deepseek_model(model_name):
        return "deepseek"
    if is_anthropic_model(model_name):
        return "anthropic"
    if is_google_model(model_name):
        return "google"

    raise ValueError(f"Model {model_name} not supported, please choose from openai, deepseek, anthropic, google")


def get_chat_model(model_name: str):
    if is_openai_model(model_name) or is_deepseek_model(model_name):
        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=0)

    if is_anthropic_model(model_name):
        from langchain_anthropic.chat_models import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=0)

    if is_google_model(model_name):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    raise ValueError(f"Model {model_name} not supported, please choose from openai, deepseek, anthropic, google")