# 该脚本的目的：
# 1. 收集正确的 XPath
# 2. 通过正确的 XPath 在页面中高亮显示对应的元素，然后用来判断
import pandas as pd
import os
import json
import html
from playwright.sync_api import sync_playwright

DATA_ROOT = "data/swde"


def normalize(text):
    import re

    # print(text_list)
    text = html.unescape(text)
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
    # text = text.replace("  ", " ")
    # text = re.sub(r"\s+", "", text)
    return text.strip()


def label_xpath_row(page, row):
    full_url = os.path.abspath(os.path.join(DATA_ROOT, row["file_path"]))

    page.goto(f"file://{full_url}")

    # highlight ground truth
    ground_truth = json.loads(row["attributes"])
    result = {}
    prev_elements = []
    for k, v in ground_truth.items():
        for ele in prev_elements:
            ele.evaluate("node => node.style.backgroundColor = ''")

        for text in v:
            normalized_text = normalize(text)
            count = page.get_by_text(normalized_text).count()
            for i in range(count):
                ele = page.get_by_text(normalized_text).nth(i)
                if ele.is_visible():
                    ele.evaluate("node => node.style.backgroundColor = 'yellow'")
                    prev_elements.append(ele)

        # scroll first into view
        if len(prev_elements) > 0:
            prev_elements[0].evaluate("node => node.scrollIntoView()")

        target_texts = ", ".join([normalize(x) for x in v])
        print(f"正在标记字段：{k}，目标文本：{target_texts}")
        xpaths = input("XPath：")
        result[k] = [x.strip() for x in xpaths.split(",")]
    return result


def handle_block(route):
    if route.request.url.endswith(".htm") or route.request.url.endswith(".html"):
        return route.continue_()
    route.abort()


def label_xpath():
    df = pd.read_csv("data/swde_token_stats.csv")
    if not os.path.exists("data/swde_xpath_labels.csv"):
        done_df = pd.DataFrame(columns=["xpath", "category", "site", "page_id"])
    else:
        done_df = pd.read_csv("data/swde_xpath_labels.csv")

    candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        context = browser.new_context(java_script_enabled=False, no_viewport=True)
        # context = browser.new_context(no_viewport=True)
        page = context.new_page()
        page.route("**/*", handle_block)
        page.set_default_timeout(0)

        try:
            rows = []
            for category, site in candidates:
                df_subset = df[(df["category"] == category) & (df["site"] == site)]
                df_subset = df_subset.sample(3, random_state=0)

                for _, row in df_subset.iterrows():
                    # skip if already labeled
                    if (
                        done_df[
                            (done_df["category"] == category)
                            & (done_df["site"] == site)
                            & (done_df["page_id"] == row["page_id"])
                        ].shape[0]
                        > 0
                    ):
                        continue

                    print(f"正在标记：{category} - {site} - {row['page_id']}")
                    xpath_dict = label_xpath_row(page, row)
                    rows.append(
                        {
                            "xpath": json.dumps(xpath_dict),
                            "category": category,
                            "site": site,
                            "page_id": row["page_id"],
                        }
                    )
        except Exception as e:
            print(e)
        finally:
            result_df = pd.DataFrame(rows)
            result_df = pd.concat([done_df, result_df])
            result_df.to_csv("data/swde_xpath_labels.csv", index=False)


if __name__ == "__main__":
    label_xpath()
