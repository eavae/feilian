from inscriptis import get_text, ParserConfig


def convert_html_to_text(html: str) -> str:
    text = get_text(html, ParserConfig(display_links=True, display_anchors=True))

    # remove leading and trailing whitespaces
    texts = text.split("\n")
    texts = [t.strip() for t in texts if t.strip()]

    # replace \n...\n with \n
    text = "\n".join(texts)
    text = text.replace("\n\n", "\n")

    return text
