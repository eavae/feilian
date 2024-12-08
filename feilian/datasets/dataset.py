import abc
import os
import json
import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass
from feilian.text_tools import normalize_text


class Dataset(abc.ABC):
    data_folder: str
    eval_sample_size: int

    def __init__(self, data_folder: str, eval_sample_size: int) -> None:
        self.data_folder = data_folder
        self.eval_sample_size = eval_sample_size

    def download(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def to_seed(self) -> "SeedDataset":
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


@dataclass
class Seed:
    id: str
    query: str
    htmls: List[str]
    ground_truth: List[dict]


@dataclass
class Sample:
    id: str
    html: str
    ground_truth: str


class SeedDataset:
    """
    A SeedDataset must have index, html, query columns.
    - index: unique identifier for each seed
    - html: html content of the seed
    - query: the query for the seed
    """

    def __init__(self, df: pd.DataFrame) -> None:
        assert "id" in df.columns, "id column is required"
        assert "html" in df.columns, "html column is required"
        assert "query" in df.columns, "query column is required"
        assert "ground_truth" in df.columns, "ground_truth column is required"

        self.data_df = df

    def __len__(self):
        """Return the number of groups."""
        groups = self.data_df["id"].drop_duplicates()
        return len(groups)

    def __getitem__(self, idx: str) -> Seed:
        df = self.data_df[self.data_df["id"] == idx]
        return Seed(
            id=idx,
            query=df["query"].iloc[0],
            htmls=df["html"].tolist(),
            ground_truth=[json.loads(x) for x in df["ground_truth"].tolist()],
        )

    def __iter__(self):
        for idx in self.data_df["id"].drop_duplicates():
            yield idx, self[idx]


class SWDE(Dataset):
    def __init__(self, data_folder: str, eval_sample_size: int, seed: int) -> None:
        super().__init__(data_folder, eval_sample_size)
        self.random_state = np.random.RandomState(seed)

        df = self.read_data()
        seed_df = df.groupby(["category", "site"]).apply(
            lambda x: x.sample(
                n=3,
                random_state=self.random_state,
            )
        )
        drop_indices = set(i[2] for i in seed_df.index)
        seed_df.reset_index(drop=True, inplace=True)
        self.seed_df = seed_df
        df: pd.DataFrame = df.drop(drop_indices)
        df.reset_index(drop=True, inplace=True)

        # sample in group category, site
        df = (
            df.groupby(["category", "site"])
            .apply(
                lambda x: x.sample(
                    n=self.eval_sample_size,
                    random_state=self.random_state,
                ),
            )
            .reset_index(drop=True)
        )
        self.df = df

        # sample in group category, site
        df = (
            df.groupby(["category", "site"])
            .apply(
                lambda x: x.sample(
                    n=self.eval_sample_size,
                    replace=True,
                    random_state=self.random_state,
                ),
            )
            .reset_index(drop=True)
        )
        self.df = df

    @property
    def name(self):
        return "SWDE"

    @property
    def categories(self):
        # read categories
        categories = []
        for path in (Path(self.data_folder) / "sourceCode" / "sourceCode").glob("*"):
            dirname = path.name
            # if all lowercase name, then it is a category
            if dirname.islower() and "." not in dirname and dirname != "groundtruth":
                categories.append(dirname)
        return categories

    def to_seed(self):
        # populate html and query
        ids = []
        htmls = []
        quries = []
        for i, row in self.seed_df.iterrows():
            category = row["category"]
            site = row["site"]
            page_id = row["page_id"]
            pages = row["pages"]

            html_path = os.path.join(self.data_folder, "sourceCode/sourceCode", category, f"{category}-{site}({pages})/{page_id}.htm")
            with open(html_path, "r") as f:
                html = f.read()
            htmls.append(html)

            query_path = os.path.join("datasets/swde/questions_en/", f"{category}_{site}.txt")
            with open(query_path, "r") as f:
                query = f.read()
            quries.append(query)

            ids.append(f"{category}_{site}")

        seed_df = pd.DataFrame(
            {
                "id": ids,
                "html": htmls,
                "query": quries,
                "ground_truth": self.seed_df["ground_truth"],
            }
        )
        return SeedDataset(seed_df)

    def get_site_pages(self, category):
        sites = []
        source_folder = os.path.join(self.data_folder, f"sourceCode/sourceCode/{category}")
        for path in os.listdir(source_folder):
            path = path.split("-")[1]
            site = re.search(r"\w+", path).group()
            pages = re.search(r"\d+", path).group()
            sites.append((site, pages))
        return sites

    def _get_ground_truth(self, category, site):
        ground_truth_folder = os.path.join(self.data_folder, f"sourceCode/sourceCode/groundtruth/{category}")

        df = pd.DataFrame()
        for file_path in Path(ground_truth_folder).rglob(f"{category}-{site}-*.txt"):
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

        # convert to json string
        df = df.fillna(json.dumps([]))
        df = df.apply(
            lambda x: (
                json.dumps(
                    {k: [normalize_text(t) for t in json.loads(v)] for k, v in x.to_dict().items()},
                    ensure_ascii=False,
                )
                if x is not None
                else x
            ),
            axis=1,
        ).to_frame(name="ground_truth")

        df["category"] = category
        df["site"] = site
        df.reset_index(inplace=True)
        return df

    def read_data(self):
        dfs = []
        for category in self.categories:
            for site, pages in self.get_site_pages(category):
                df = self._get_ground_truth(category, site)
                df["pages"] = pages
                dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        return df

    def download(self):
        raise NotImplementedError

    def __len__(self):
        """how many groups in the dataset"""
        groups = self.df["category", "site"].drop_duplicates()
        return len(groups)

    def __getitem__(self, idx: str) -> List[Sample]:
        """
        using the group index, return a list of instances
        """
        category, site = idx.split("_")
        df = self.df[(self.df["category"] == category) & (self.df["site"] == site)]

        # populate html and ground_truth
        instances = []
        for i, row in df.iterrows():
            page_id = row["page_id"]
            pages = row["pages"]

            html_path = os.path.join(self.data_folder, "sourceCode/sourceCode", category, f"{category}-{site}({pages})/{page_id}.htm")
            with open(html_path, "r") as f:
                html = f.read()

            ground_truth = json.loads(row["ground_truth"])
            instances.append(
                Sample(
                    id=f"{category}_{site}_{page_id}",
                    html=html,
                    ground_truth=ground_truth,
                )
            )
        return instances


class SWDEExpanded(Dataset):
    def __init__(self, data_folder: str, eval_sample_size: int, seed: int, swde_data_folder="../swde") -> None:
        super().__init__(data_folder, eval_sample_size)
        self.random_state = np.random.RandomState(seed)
        self.swde_data_folder = os.path.join(data_folder, swde_data_folder)

        df = self.read_data()
        seed_df = df.groupby(["category", "site"]).apply(
            lambda x: x.sample(
                n=3,
                random_state=self.random_state,
            )
        )
        drop_indices = set(i[2] for i in seed_df.index)
        seed_df.reset_index(drop=True, inplace=True)
        self.seed_df = seed_df
        df.drop(drop_indices, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # sample in group category, site
        df = (
            df.groupby(["category", "site"])
            .apply(
                lambda x: x.sample(
                    n=self.eval_sample_size,
                    random_state=self.random_state,
                ),
            )
            .reset_index(drop=True)
        )
        self.df = df

    @property
    def name(self):
        return "SWDE_Expanded"

    @property
    def categories(self):
        # read categories
        categories = []
        for path in Path(self.data_folder).glob("*"):
            if path.is_dir():
                categories.append(path.name)
        return categories

    def get_site_pages(self, category):
        sites = []
        source_folder = os.path.join(self.data_folder, category)
        for path in os.listdir(source_folder):
            path = path.split("-")[1]
            site = re.search(r"\w+", path).group()
            pages = re.search(r"\d+", path).group()
            sites.append((site, pages))
        return sites

    def read_data(self):
        dfs = []
        for category in self.categories:
            for site, pages in self.get_site_pages(category):
                records = []
                json_path = os.path.join(self.data_folder, category, f"{category}-{site}({pages}).json")
                data = json.load(open(json_path, "r"))

                fields = set()
                for page_id, ground_truth in data.items():
                    # filter ground truth
                    new_ground_truth = {}
                    for k, v in ground_truth.items():
                        if v and not k.startswith("."):
                            if k.endswith(":"):
                                k = k[:-1].strip()
                            new_ground_truth[k] = v
                            fields.add(k)

                    records.append(
                        {
                            "category": category,
                            "site": site,
                            "page_id": page_id,
                            "ground_truth": json.dumps(new_ground_truth, ensure_ascii=False),
                        }
                    )
                fields = sorted(list(fields))
                query = "Please extract the following fields: " + ", ".join([f"`{field}` of the {category}" for field in fields])

                df = pd.DataFrame(records)
                df["query"] = query
                df['pages'] = pages
                dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        return df

    def __getitem__(self, idx):
        """
        using the group index, return a list of instances
        """
        category, site = idx.split("_")
        df = self.df[(self.df["category"] == category) & (self.df["site"] == site)]

        # populate html and ground_truth
        instances = []
        for i, row in df.iterrows():
            page_id = row["page_id"]
            pages = row["pages"]

            html_path = os.path.join(self.swde_data_folder, "sourceCode/sourceCode", category, f"{category}-{site}({pages})/{page_id}")
            ground_truth = json.loads(row["ground_truth"])
            instances.append(
                Sample(
                    id=f"{category}_{site}_{page_id}",
                    html=open(html_path, "r").read(),
                    ground_truth=ground_truth,
                )
            )
        return instances

    def __len__(self):
        """how many groups in the dataset"""
        groups = self.df["category", "site"].drop_duplicates()
        return len(groups)

    def to_seed(self):
        # populate html and query
        ids = []
        htmls = []
        for i, row in self.seed_df.iterrows():
            category = row["category"]
            site = row["site"]
            page_id = row["page_id"]
            pages = row["pages"]

            html_path = os.path.join(self.swde_data_folder, "sourceCode/sourceCode", category, f"{category}-{site}({pages})/{page_id}")
            with open(html_path, "r") as f:
                html = f.read()
            htmls.append(html)

            ids.append(f"{category}_{site}")

        seed_df = pd.DataFrame(
            {
                "id": ids,
                "html": htmls,
                "query": self.seed_df["query"],
                "ground_truth": self.seed_df["ground_truth"],
            }
        )
        return SeedDataset(seed_df)
