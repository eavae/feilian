import os
import py7zr
import re
import pandas as pd
import json
import bs4
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from minify_html import minify
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from feilian.soup_tools import get_node_contain_text, get_common_ancestor, clean_html
from feilian.prompts import QUESTION_CONSTRUCTION_EN


def remove_hidden_files(src_folder, exclude_files=None):

    if not exclude_files:
        exclude_files = {
            ".gitkeep",
            ".gitignore",
        }

    for root, dirs, files in os.walk(src_folder, followlinks=True):
        for file in files:
            if file in exclude_files:
                continue

            if file.startswith("."):
                os.remove(os.path.join(root, file))


def _extract_7z_recursive(dir_or_file_path):
    if isinstance(dir_or_file_path, list):
        zip_files = dir_or_file_path
    elif isinstance(dir_or_file_path, str) and os.path.isdir(dir_or_file_path):
        zip_files = Path(dir_or_file_path).rglob("*.7z")
    else:
        zip_files = [dir_or_file_path]

    for zip_file in tqdm(zip_files, desc="Extracting 7z files"):
        zip_file = Path(zip_file)
        file_name = zip_file.name
        if file_name.startswith("."):
            continue

        if zip_file.is_dir():
            continue

        with py7zr.SevenZipFile(zip_file, "r") as archive:
            folder = zip_file.parent
            archive.extractall(folder)


def unzip_all_recursive(src_dir):
    zip_files = defaultdict(list)
    for root, dirs, files in os.walk(src_dir, followlinks=True):
        for file in files:
            if file.endswith(".7z"):
                zip_files["7z"].append(os.path.join(root, file))
    _extract_7z_recursive(zip_files["7z"])


def swde__read_ground_truth(root_folder: str, category: str, site: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for file_path in Path(root_folder).rglob(f"{category}-{site}-*.txt"):
        attr_name = file_path.stem.split("-")[-1]
        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = lines[2:]
            lines = [x.strip() for x in lines if x.strip()]
            lines = [x.split("\t") for x in lines]
            lines = [x for x in lines if x[2] != "<NULL>"]

            records = []
            for line in lines:
                page_id = line[0]
                values = line[2:]
                records.append((page_id, json.dumps(values, ensure_ascii=False)))

            _df = pd.DataFrame(
                records,
                columns=["page_id", attr_name],
                dtype=str,
            )
            _df.set_index("page_id", inplace=True)

            if df.empty:
                df = _df
            else:
                df = df.merge(_df, how="outer", left_index=True, right_index=True)

    return df.fillna(json.dumps([]))


# SWDE Dataset
def swde__convert_to_parquet(root_folder: str, save_to: str):
    ground_truth_folder = os.path.join(root_folder, "sourceCode/sourceCode/groundtruth")
    html_folder = "sourceCode/sourceCode"
    categories = []

    for x in os.listdir(ground_truth_folder):
        file_path = os.path.join(ground_truth_folder, x)
        if Path(file_path).is_dir() and not file_path.startswith("."):
            categories.append(x)

    records = []
    for category in categories:
        folders = os.listdir(os.path.join(root_folder, html_folder, category))
        for folder in folders:
            part = folder.split("-")[1]
            site = re.search(r"\w+", part).group()

            df = swde__read_ground_truth(
                os.path.join(ground_truth_folder, category),
                category,
                site,
            )
            for page_id, row in df.iterrows():
                file_path = os.path.join(
                    html_folder,
                    category,
                    folder,
                    f"{page_id}.htm",
                )
                attributes = row.to_dict()
                attributes = {
                    k: json.loads(v) for k, v in attributes.items() if v != "[]"
                }
                records.append(
                    (
                        category,
                        site,
                        page_id,
                        file_path,
                        json.dumps(attributes, ensure_ascii=False),
                    )
                )

    df = pd.DataFrame(
        records, columns=["category", "site", "page_id", "file_path", "attributes"]
    )
    df.to_parquet(save_to)


def swde__construct_question(file_path: str):
    """为SWDE数据集中的每个类别构造问题，比如：auto

    Args:
        file_path (str): _description_
    """

    # construct question using langchain
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.1)
    template = ChatPromptTemplate.from_messages([("human", QUESTION_CONSTRUCTION_EN)])
    chain = template | llm

    df = pd.read_csv(file_path, index_col=0)
    for group, frame in df.groupby(["category", "site"]):
        save_to = f"datasets/swde/questions_en/{'_'.join(group)}.txt"
        if os.path.exists(save_to):
            continue

        # sort by cleaned_tokens
        frame = frame.sort_values("cleaned_tokens", ascending=False)
        row = frame.iloc[0]

        # read html
        html_file = os.path.join("data/swde", row["file_path"])
        raw_html = open(html_file, "r").read()
        soup = bs4.BeautifulSoup(raw_html, "html5lib")

        # find target nodes
        target_texts = []
        for values in json.loads(row["attributes"]).values():
            target_texts.extend(values)
        nodes = [get_node_contain_text(soup, x) for x in target_texts]
        nodes = [x for x in nodes if x]

        # get common ancestor
        node = get_common_ancestor(nodes)
        cleaned_node = clean_html(node)
        cleaned_html = minify(cleaned_node.prettify().strip())

        question = chain.invoke(
            {
                "answer_str": row["attributes"],
                "context_str": cleaned_html,
            }
        )

        with open(save_to, "w") as f:
            f.write(question.content)
            f.write("\n")
    pass


if __name__ == "__main__":
    swde__construct_question("swde_token_stats.csv")
