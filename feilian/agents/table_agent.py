import pandas as pd
import json
import bs4
import html5lib
from minify_html import minify
from langchain_openai import ChatOpenAI

from feilian.prompts import TABLE_EXTRACTION_PROMPT_CN, TABLE_EXTRACTION_PROMPT_HISTORY
from feilian.etree_tools import clean_html, to_string


def create_table_corr_chain():
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.1,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return TABLE_EXTRACTION_PROMPT_CN | llm


# node = {
#     "children": []
# }
def deep_first_traversal(nodes: list):
    for node in nodes:
        yield from deep_first_traversal(node["children"])
        yield node


if __name__ == "__main__":
    df = pd.read_csv("swde_extracted_tables.csv")
    df = df[(df["category"] == "university") & (df["site"] == "embark")]
    df = df.sample(8)

    chain = create_table_corr_chain()

    results = []
    for i, row in df.iterrows():
        remove_xpaths = []
        target_xpaths = []

        remove_tables = []
        tables = json.loads(row["tables"])
        for table in deep_first_traversal(tables):
            for remove_table in remove_tables:
                table["content"] = table["content"].replace(remove_table, "")

            tree = html5lib.parseFragment(
                table["content"],
                treebuilder="lxml",
                namespaceHTMLElements=False,
            )
            if isinstance(tree, list):
                tree = tree[0]

            clean_html(tree)
            cleaned_table = minify(to_string(tree))
            if not cleaned_table or not tree.getchildren():
                remove_tables.append(table["content"])
                remove_xpaths.append(table["xpath"])
                continue

            question = open(
                "datasets/swde/questions_cn/university_embark.txt", "r"
            ).read()

            response = chain.invoke(
                {
                    "chat_history": TABLE_EXTRACTION_PROMPT_HISTORY,
                    "table": cleaned_table,
                    "question": question,
                }
            )
            json_response = json.loads(response.content)
            if set(json_response.keys()) - {"_thought"}:
                remove_tables.append(table["content"])
                target_xpaths.append(table["xpath"])

        results.append(
            {
                "category": row["category"],
                "site": row["site"],
                "page_id": row["page_id"],
                "remove_xpaths": remove_xpaths,
                "target_xpaths": target_xpaths,
            }
        )

    with open("swde_table_test.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
