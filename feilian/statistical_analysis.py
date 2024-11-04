import tiktoken
import pandas as pd
import numpy as np
import os
import bs4
import json
import html
import warnings
from minify_html import minify
from tqdm import tqdm
from typing import List
from html5lib.constants import DataLossWarning

from feilian.soup_tools import (
    clean_html as soup_clean_html,
    get_structure,
    prune_by_structure,
    extract_tables_recursive,
    get_tables_depth,
    get_tables_count,
    get_tables_max_width,
    get_tables_width,
)
from feilian.etree_token_stats import extract_fragments_by_weight
from feilian.etree_tools import parse_html, clean_html as etree_clean_html, to_string

SWDE_DATA_ROOT = "data/swde"

tqdm.pandas()

encoder = tiktoken.encoding_for_model("gpt-4")

warnings.filterwarnings(action="ignore", category=DataLossWarning, module=r"html5lib")


def tokenizer(text):
    if not text:
        return 0
    return len(encoder.encode(text))


def swde__stats_token_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    html = open(html_file_path).read()
    tree = parse_html(html)
    etree_clean_html(tree, True)
    cleaned_html = minify(to_string(tree), keep_closing_tags=True)

    row["raw_tokens"] = tokenizer(html)
    row["cleaned_tokens"] = tokenizer(cleaned_html)

    return row


def _bin_fn(x):
    return x // 4000 * 4000


def swde__plot_token_dist(
    dataset_file_path: str,
    target_col: str = "cleaned_tokens",
    sample_size: int = 128,
):
    import plotly.graph_objects as go
    import plotly.express as px

    # read & sample
    df = pd.read_csv(dataset_file_path, index_col=0)
    df = df.groupby(["category", "site"]).apply(
        lambda x: x.sample(min(len(x), sample_size), replace=False)
    )
    df.reset_index(drop=True, inplace=True)

    # count tokens
    df = df.apply(swde__stats_token_row, axis=1)
    df["token_cat"] = df["raw_tokens"].apply(_bin_fn)
    group_df = (
        df.groupby("token_cat")["raw_tokens", "cleaned_tokens"].mean().astype(int)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=group_df.index,
            y=group_df["raw_tokens"],
            name="tokens",
            marker_color=px.colors.qualitative.D3[0],
        )
    )
    fig.add_trace(
        go.Bar(
            x=group_df.index,
            y=group_df["cleaned_tokens"],
            name="after sanitization",
            marker_color=px.colors.qualitative.D3[1],
        )
    )

    fig.update_layout(
        # title_text=f"Information Extraction Sanitizer Analyse (Evenly Sampled {len(df)} from SWDE)",  # title of plot
        title_x=0.5,  # title x location
        xaxis_title_text="binned at 4k intervals",  # xaxis label
        yaxis_title_text="mean",  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
        xaxis=dict(
            tickmode="array",
            tickvals=group_df.index,
            ticktext=[f"{x // 1000}k" for x in group_df.index],
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin={"t": 4, "l": 4, "b": 4, "r": 4},
    )
    # fig.update_yaxes(type="log")
    fig.update_xaxes(title_standoff=8)
    fig.update_yaxes(title_standoff=8)
    fig.write_image("fig1.eps")
    fig.show()


def read_and_clean_html(file_path: str):
    with open(file_path, "r") as f:
        html_content = f.read()
    soup = bs4.BeautifulSoup(html_content, "html5lib")
    cleaned_soup = soup_clean_html(soup)
    cleaned_html = minify(cleaned_soup.prettify().strip(), keep_closing_tags=True)

    return html_content, cleaned_html


def read_and_structure_html(file_path: str):
    html_file_path = os.path.join(SWDE_DATA_ROOT, file_path)
    html_text = open(html_file_path, "r", encoding="utf-8").read()
    soup = bs4.BeautifulSoup(html_text, "html5lib")
    soup = soup_clean_html(soup)

    structure = get_structure(html_text)
    prune_by_structure(soup, structure)

    cleaned_html = minify(soup.prettify().strip(), keep_closing_tags=True)

    return structure.prettify().strip(), cleaned_html


def swde__test_semantic_pruning_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    _, cleaned_html = read_and_clean_html(html_file_path)

    values = set()
    attributes = json.loads(row["attributes"])
    for xs in attributes.values():
        for x in xs:
            x = x.strip()
            values.add(x)

    matches = []
    for value in values:
        if value in cleaned_html:
            matches.append(True)
            continue

        if html.unescape(value) in cleaned_html:
            matches.append(True)
            continue

        matches.append(False)

    row["semantic_pruning"] = all(matches)
    return row


def swde__test_semantic_pruning(dataset_file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=1)

    df = pd.read_csv(dataset_file_path)
    df = df.parallel_apply(swde__test_semantic_pruning_row, axis=1)
    # df = df.progress_apply(swde__test_semantic_pruning_row, axis=1)

    failed_case = df[df["semantic_pruning"] == False]  # noqa
    print(f"Failed cases: {len(failed_case)}")
    failed_case.to_csv("semantic_pruning_failed_cases.csv")


def swde__stats_structure_row(row):
    _, cleaned_html = read_and_structure_html(row["file_path"])
    row["structure_tokens"] = len(encoder.encode(cleaned_html))

    values = set()
    attributes = json.loads(row["attributes"])
    for xs in attributes.values():
        for x in xs:
            x = x.strip()
            values.add(x)

    matches = []
    for value in values:
        if value in cleaned_html:
            matches.append(True)
            continue

        if html.unescape(value) in cleaned_html:
            matches.append(True)
            continue

        matches.append(False)

    row["match_count"] = sum(matches)
    row["total"] = len(matches)

    return row


def swde__stats_structure_pruning(dataset_file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=4)

    df = pd.read_csv(dataset_file_path, index_col=0)

    # remove index
    df.reset_index(drop=True, inplace=True)

    df = df.parallel_apply(swde__stats_structure_row, axis=1)
    df.to_csv("swde_token_stats_with_structure.csv")


def swde__compression_ratio(
    dataset_file_path: str, group_by, baseline_col: str, target_cols: List[str]
):

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_csv(dataset_file_path, index_col=0)

    pairs = df[group_by].groupby(group_by).nunique().index
    n_cols = 8
    n_rows = len(pairs) // n_cols + 1

    df = df.groupby(group_by).mean().reset_index()
    for target_col in target_cols:
        df[f"{target_col}_compression_ratio"] = df[baseline_col] / df[target_col]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{pair}" for pair in pairs],
    )

    for i, pair in enumerate(pairs):
        row, col = i // n_cols + 1, i % n_cols + 1
        pair_df = df[(df[group_by] == pair).all(axis=1)]

        # melt and renaming
        bar_df = pair_df.melt(
            id_vars=group_by,
            value_vars=[baseline_col]
            + [f"{target_col}_compression_ratio" for target_col in target_cols],
        )
        bar_df.loc[bar_df["variable"] == baseline_col, "variable"] = "baseline"
        bar_df.loc[bar_df["variable"] == "baseline", "value"] = 1
        for target_col in target_cols:
            bar_df.loc[
                bar_df["variable"] == f"{target_col}_compression_ratio", "variable"
            ] = target_col

        fig.add_trace(
            go.Bar(
                x=bar_df["variable"],
                y=bar_df["value"],
                name=f"{pair}",
            ),
            row=row,
            col=col,
        )
    fig.update_layout(height=2048, title_text="Compression Ratio")
    fig.show()


def swde__plot_failure(dataset_file_path: str, group_by):

    import plotly.express as px

    df = pd.read_csv(dataset_file_path, index_col=0)

    df = df[group_by + ["match_count", "total"]].groupby(group_by).sum().reset_index()
    df["failed_count"] = df["total"] - df["match_count"]
    df["category_site"] = df[group_by].apply(lambda x: f"{x[0]}_{x[1]}", axis=1)

    fail_rate_df = df[["category_site", "failed_count", "total"]].copy()
    fail_rate_df["failed_rate"] = fail_rate_df["failed_count"] / (fail_rate_df["total"])
    fail_rate_df = fail_rate_df.sort_values("failed_rate", ascending=False)

    # melt, x=category_site, y=count, color=failed
    df = df.melt(
        id_vars=["category_site"],
        value_vars=["failed_count", "match_count"],
    )

    # plot bar chart
    fig = px.bar(
        df,
        x="category_site",
        y="value",
        color="variable",
        title="Failed Cases",
    )

    fig.update_layout(
        xaxis_title="Category_Site",
        yaxis_title="Count",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=fail_rate_df["category_site"])
    fig.show()
    pass


def swde__extract_table_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    html_content = open(html_file_path, "r").read()
    soup = bs4.BeautifulSoup(html_content, "html5lib")
    tables = extract_tables_recursive(soup)

    row["tables"] = json.dumps(tables)

    return row


def swde__extract_tables(file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=6)

    df = pd.read_csv(file_path, index_col=0)

    df = df.parallel_apply(swde__extract_table_row, axis=1)
    df.to_csv("swde_extracted_tables.csv")


def swde__table_correlation_row(row):
    tables = json.loads(row["tables"])
    fns = {
        "depth": get_tables_depth,
        "count": get_tables_count,
        "max_width": get_tables_max_width,
        "width": get_tables_width,
    }
    for k, fn in fns.items():
        row[k] = fn(tables)
    return row


def swde_table_correlation_analyse(file_path: str):
    """结论：没有明显的相关性

    Args:
        file_path (str): _description_
    """
    # from pandarallel import pandarallel
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # pandarallel.initialize(progress_bar=True, nb_workers=4)

    group_by = ["category", "site"]

    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(swde__table_correlation_row, axis=1)

    df["match_rate"] = df["match_count"] / df["total"]
    df = df[df["match_rate"] < 1]

    # plot correlation
    pairs = df[group_by].groupby(group_by).nunique().index
    n_cols = 8
    n_rows = len(pairs) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{pair}" for pair in pairs],
    )

    for i, pair in enumerate(pairs):
        row, col = i // n_cols + 1, i % n_cols + 1
        pair_df = df[(df[group_by] == pair).all(axis=1)]

        # correlations, match_rate, depth, count, max_width, width
        corr = pair_df[["match_rate", "depth", "count", "max_width", "width"]].corr()
        corr.fillna(0, inplace=True)

        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
                showscale=False,
            ),
            row=row,
            col=col,
        )
    fig.show()


def swde__stats_parallel_pruning_row(row, until_html_tokens, max_text_tokens):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    html = open(html_file_path).read()

    tree = parse_html(html)
    etree_clean_html(tree)
    initial_tokens = tokenizer(minify(to_string(tree), keep_closing_tags=True))
    fragment_tokens = []

    for xpath in extract_fragments_by_weight(
        tree, tokenizer, until_html_tokens, max_text_tokens
    ):
        for node in tree.xpath(xpath):
            node.clear()
            node.text = ""
        tokens = tokenizer(minify(to_string(tree), keep_closing_tags=True))
        fragment_tokens.append(initial_tokens - tokens)
        initial_tokens = tokens

    row["fragment_tokens"] = json.dumps(fragment_tokens)
    row["finial_tokens"] = initial_tokens

    return row


def swde__stats_parallel_pruning():
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=12)

    df = pd.read_csv("data/swde_token_stats.csv")
    # df = df.iloc[0:1]

    # sample 16 per group, group by category, site
    df = df.groupby(["category", "site"]).apply(
        lambda x: x.sample(
            min(len(x), 16),
            replace=False,
            random_state=0,
        )
    )

    # remove index
    df.reset_index(drop=True, inplace=True)

    dfs = []
    experiments = [
        (1024, 256),
        (1024 * 2, 256),
        (1024 * 3, 256),
        (1024, 512),
        (1024 * 2, 512),
        (1024 * 3, 512),
        (1024, 1024),
        (1024 * 2, 1024),
        (1024 * 3, 1024),
    ]
    for until_html_tokens, max_text_tokens in experiments:
        exp_df = df.parallel_apply(
            swde__stats_parallel_pruning_row,
            axis=1,
            args=(until_html_tokens, max_text_tokens),
        )
        exp_df["until_html_tokens"] = until_html_tokens
        exp_df["max_text_tokens"] = max_text_tokens
        dfs.append(exp_df)
    exp_df = pd.concat(dfs)

    exp_df.to_csv("data/swde_parallel_pruning_exp.csv")


# 绘制 html 片段数量统计
# 散点图
def swde__plot_fragment_count_stats():

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_csv("data/swde_parallel_pruning_exp.csv")
    df["fragment_tokens"] = df["fragment_tokens"].apply(lambda x: json.loads(x))
    df["fragment_count"] = df["fragment_tokens"].apply(lambda x: len(x))

    experiments = [
        (1024, 256),
        (1024 * 2, 256),
        (1024 * 3, 256),
        (1024, 512),
        (1024 * 2, 512),
        (1024 * 3, 512),
        (1024, 1024),
        (1024 * 2, 1024),
        (1024 * 3, 1024),
    ]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            f"until_html_tokens={until_html_tokens}, max_text_tokens={max_text_tokens}"
            for (until_html_tokens, max_text_tokens) in experiments
        ],
    )
    titles = []
    for i, (until_html_tokens, max_text_tokens) in enumerate(experiments):
        row, col = i // 3 + 1, i % 3 + 1
        exp_df = df[
            (df["until_html_tokens"] == until_html_tokens)
            & (df["max_text_tokens"] == max_text_tokens)
        ]
        fig.add_trace(
            go.Scatter(
                x=exp_df["cleaned_tokens"],
                y=exp_df["fragment_count"],
                mode="markers",
                name=f"{until_html_tokens}-{max_text_tokens}",
            ),
            row=row,
            col=col,
        )
        titles.append(f"{until_html_tokens}-{max_text_tokens}")
    # fig.for_each_annotation(lambda a: a.update(text=))
    fig.show()
    pass


# 绘制 html 片段中 token 数的统计
# 绘制 4 张图，分别是 1024, 2048, 3072, 4096 的片段数量统计
# 横坐标是 cleaned_tokens 的分桶
# 纵坐标是 fragment_tokens 的分布（均值，95% 分位数，5% 分位数）
def swde__plot_fragment_token_stats():
    import plotly.graph_objects as go

    df = pd.read_csv("data/swde_parallel_pruning.csv")
    df["cleaned_tokens"] = df["cleaned_tokens"].apply(lambda x: x // 1000 * 1000)
    df["removed_tokens"] = df["removed_tokens"].apply(lambda x: json.loads(x))

    fig = go.Figure()

    # flatten fragment_tokens
    limit_df = df.explode("removed_tokens")
    limit_df["removed_tokens"] = limit_df["removed_tokens"].fillna(0)
    limit_df["removed_tokens"] = limit_df["removed_tokens"].astype(int)

    # group by cleaned_tokens
    group_df = limit_df.groupby("cleaned_tokens")["removed_tokens"].apply(
        lambda x: pd.Series(x).describe(percentiles=[0.05, 0.95])
    )
    group_df = group_df.unstack().reset_index()

    fig.add_trace(
        go.Scatter(
            x=group_df["cleaned_tokens"],
            y=group_df["mean"],
            name="mean",
            mode="markers",
            showlegend=False,
            marker=dict(color="blue", size=np.log(group_df["count"]) * 4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group_df["cleaned_tokens"],
            y=group_df["5%"],
            name="5%",
            mode="markers",
            showlegend=False,
            marker=dict(color="blue", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group_df["cleaned_tokens"],
            y=group_df["95%"],
            name="95%",
            mode="markers",
            showlegend=False,
            marker=dict(color="blue", size=8),
        )
    )

    fig.show()
    pass


if __name__ == "__main__":
    swde__plot_token_dist("data/swde_token_stats.csv")
