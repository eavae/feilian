import pandas as pd
import json
import os
import html5lib
from minify_html import minify
from feilian.etree_tools import (
    clean_html,
    to_string,
    pre_order_traversal,
    prune_by_xpath,
    deduplicate_xpath,
)

if __name__ == "__main__":
    dataset_root = "data/swde"

    df = pd.read_csv("swde_token_stats.csv")
    items = json.loads(open("swde_table_test.json").read())
    for item in items:
        category = item["category"]
        site = item["site"]
        page_id = item["page_id"]

        row = df[
            (df["category"] == category)
            & (df["site"] == site)
            & (df["page_id"] == page_id)
        ].iloc[0]

        raw_html = open(os.path.join(dataset_root, row["file_path"]), "r").read()
        tree = html5lib.parse(raw_html, treebuilder="lxml", namespaceHTMLElements=False)
        clean_html(tree)
        pre_order_traversal(
            tree,
            lambda ele, xpath: prune_by_xpath(
                ele,
                xpath,
                includes=deduplicate_xpath(item["target_xpaths"]),
                excludes=deduplicate_xpath(item["remove_xpaths"]),
            ),
        )
        cleaned_html = minify(to_string(tree))

        name = tree.xpath("//h2/text()")[0].strip()
        phone = tree.xpath(
            "//span[@class='label' and text()='Phone:']/following-sibling::span[@class='data']/text()"
        )[0].strip()
        school_type = tree.xpath(
            "//td[@class='label' and text()='School Type:']/following-sibling::td[@class='data']/text()"
        )[0].strip()
        website = tree.xpath(
            "//span[@class='label' and text()='Website:']/following-sibling::span[@class='data']/a/@href"
        )[0].strip()
        pass
