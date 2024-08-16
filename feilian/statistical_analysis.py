import tiktoken
import pandas as pd
import os
import bs4
import json
import html
from minify_html import minify
from tqdm import tqdm
from typing import List

from feilian.soup_tools import (
    clean_html,
    get_structure,
    prune_by_structure,
    extract_tables_recursive,
    get_tables_depth,
    get_tables_count,
    get_tables_max_width,
    get_tables_width,
)

SWDE_DATA_ROOT = "data/swde"

tqdm.pandas()

encoder = tiktoken.encoding_for_model("gpt-4")


def swde__stats_token_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    html_content, cleaned_html = read_and_clean_html(html_file_path)

    row["raw_tokens"] = len(encoder.encode(html_content))
    row["cleaned_tokens"] = len(encoder.encode(cleaned_html))
    return row


def swde__stats_token(dataset_file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=4)

    df = pd.read_parquet(dataset_file_path)

    # sample 256 per group, group by category, site
    df = df.groupby(["category", "site"]).apply(
        lambda x: x.sample(min(len(x), 256), replace=False)
    )

    # remove index
    df.reset_index(drop=True, inplace=True)

    df = df.parallel_apply(swde__stats_token_row, axis=1)
    df.to_csv("swde_token_stats.csv")


def swde__plot_token_dist(
    dataset_file_path: str,
    group_cols: list = ["category", "site"],
    target_col: str = "cleaned_tokens",
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_csv(dataset_file_path, index_col=0)
    pairs = df[group_cols].groupby(group_cols).nunique().index
    n_cols = 8
    n_rows = len(pairs) // n_cols + 1

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"{pair}" for pair in pairs]
    )

    for i, pair in enumerate(pairs):
        row, col = i // n_cols + 1, i % n_cols + 1
        pair_df = df[(df[group_cols] == pair).all(axis=1)]
        fig.add_trace(
            go.Histogram(x=pair_df[target_col], name=f"{pair}", histnorm="percent"),
            row=row,
            col=col,
        )

    fig.update_layout(height=2048)
    fig.show()


def read_and_clean_html(file_path: str):
    with open(file_path, "r") as f:
        html_content = f.read()
    soup = bs4.BeautifulSoup(html_content, "html5lib")
    cleaned_soup = clean_html(soup)
    cleaned_html = minify(cleaned_soup.prettify().strip())

    return html_content, cleaned_html


def read_and_structure_html(file_path: str):
    html_file_path = os.path.join(SWDE_DATA_ROOT, file_path)
    html_text = open(html_file_path, "r", encoding="utf-8").read()
    soup = bs4.BeautifulSoup(html_text, "html5lib")
    soup = clean_html(soup)

    structure = get_structure(html_text)
    prune_by_structure(soup, structure)

    cleaned_html = minify(soup.prettify().strip())

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


if __name__ == "__main__":
    swde_table_correlation_analyse("swde_extracted_tables.csv")

    # structure, cleaned_html = read_and_structure_html(
    #     "sourceCode/sourceCode/restaurant/restaurant-pickarestaurant(2000)/0000.htm"
    # )

    # with open("test.html", "w") as f:
    #     f.write(cleaned_html)

    # with open("structure_test.html", "w") as f:
    #     f.write(structure)
    pass
