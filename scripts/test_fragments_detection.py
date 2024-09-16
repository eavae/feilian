from feilian.agents.fragments_detection import build_graph

if __name__ == "__main__":
    import pandas as pd
    import os
    import hashlib

    # from langchain.globals import set_debug

    # set_debug(True)

    category = "auto"
    site = "msn"
    random_state = 42
    root_dir = "data/swde"
    query = open(f"datasets/swde/questions_cn/{category}_{site}.txt", "r").read()

    df = pd.read_csv("data/swde_token_stats.csv")
    df = df[(df["category"] == category) & (df["site"] == site)]
    row = df.sample(1, random_state=random_state).iloc[0]

    html = open(os.path.join(root_dir, row["file_path"])).read()
    state = {
        "id": hashlib.md5(html.encode()).hexdigest(),
        "raw_html": html,
        "query": query,
    }
    graph = build_graph()
    result = graph.invoke(state)
    pass
