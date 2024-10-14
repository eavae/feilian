import html
import re
from inscriptis import get_text, ParserConfig


def convert_html_to_text(html: str) -> str:
    text = get_text(html, ParserConfig(display_links=False, display_anchors=False))

    # remove leading and trailing whitespaces
    texts = text.split("\n")
    texts = [t.strip() for t in texts if t.strip()]

    # replace \n...\n with \n
    text = "\n".join(texts)
    text = text.replace("\n\n", "\n")

    return text


def normalize_text(text):
    # print(text_list)
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'").replace("&apos;", "'")
    text = text.replace("&#150;", "–")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#160;", " ")
    text = text.replace("&#039;", "'")
    text = text.replace("&#34;", '"')
    text = text.replace("&reg;", "®")
    text = text.replace("&rsquo;", "’")
    text = text.replace("&#8226;", "•")
    text = text.replace("&ndash;", "–")
    text = text.replace("&#x27;", "'")
    text = text.replace("&#40;", "(")
    text = text.replace("&#41;", ")")
    text = text.replace("&#47;", "/")
    text = text.replace("&#43;", "+")
    text = text.replace("&#035;", "#")
    text = text.replace("&#38;", "&")
    text = text.replace("&eacute;", "é")
    text = text.replace("&frac12;", "½")
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"  +", " ", text)
    return text.strip()
