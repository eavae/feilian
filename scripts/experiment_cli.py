# Cmd Line Tool for running experiments
# Usage: python scripts/experiment_cli.py --help [name] [output]
# args:
#   --dataset: SWDE or SWDE_Extended
#   --data_dir: data folder, default is data/swde
#   --ie_model: model name, options: deepseek-chat ...
#   --program_model: model name, options: deepseek-program ...
#   --ablation: ablation study, options: none, no_program, no_ie
#   --eval_sample_size: 32 (default), number of samples to evaluate

import argparse
import abc
import os
import json
import tqdm
import pandas as pd
from dotenv import load_dotenv
from hashlib import md5
from typing import List
from minify_html import minify
from collections import defaultdict

from feilian.models import check_model
from feilian.datasets import SWDE, Dataset, Sample
from feilian.etree_tools import normalize_text, parse_html, extract_text_by_xpath

ablation_options = {
    "wo_cue": "WITHOUT_CUE",
    "wo_gen_xpath": "WITHOUT_GEN_XPATH",
    "none": None,
}

model_mapping = {
    "ie_model": "IE_MODEL",
    "program_model": "PROGRAM_MODEL",
}


def create_dataset(dataset: str, data_dir: str, **kwargs):
    if dataset == "SWDE":
        return SWDE(data_dir, **kwargs)

    raise ValueError(f"Dataset {dataset} not supported, please choose from SWDE, SWDE_Extended")


def eval_array(predict: List[str], ground_truth: List[str]):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for value in predict:
        if value in ground_truth:
            true_positives += 1
        else:
            false_positives += 1

    for value in ground_truth:
        if value not in predict:
            false_negatives += 1

    return true_positives, false_positives, false_negatives


def eval_objects(predict, ground_truth):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for field_name in set(predict.keys()) | set(ground_truth.keys()):
        ground_truth_values = ground_truth.get(field_name, [])
        predict_values = predict.get(field_name, [])
        predict_values = [normalize_text(x) for x in predict_values]
        ground_truth_values = [normalize_text(x) for x in ground_truth_values]
        tp, fp, fn = eval_array(predict_values, ground_truth_values)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    return true_positives, false_positives, false_negatives


class Evaluator(abc.ABC):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @abc.abstractmethod
    def __call__(self, predict, ground_truth):
        pass

    def to_json(self):
        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "precision": self.tp / (self.tp + self.fp + 1e-8),
            "recall": self.tp / (self.tp + self.fn + 1e-8),
            "f1": 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-8),
        }


class IEEvaluator(Evaluator):
    def __call__(self, predict, ground_truth):
        tp, fp, fn = eval_objects(predict, ground_truth)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
        }


class XPathEvaluator(Evaluator):
    def __call__(self, samples: List[Sample], xpaths: List[dict]):
        ie_evaluator = IEEvaluator()
        for sample in samples:
            tree = parse_html(minify(sample.html, keep_closing_tags=True))
            data = {}
            for item in xpaths:
                data[item["name"]] = extract_text_by_xpath(tree, item["xpath"])[0]
            ie_evaluator(data, sample.ground_truth)
        self.tp += ie_evaluator.tp
        self.fp += ie_evaluator.fp
        self.fn += ie_evaluator.fn
        return ie_evaluator.to_json()


def run_experiment(dataset: Dataset, args, epsilon=1e-8):
    from feilian.agents.agent import build_graph, build_state

    # 1. create output folder
    os.makedirs(args.output, exist_ok=True)

    # 2. create experiment folder
    experiment_folder = os.path.join(args.output, f"{args.name}_{dataset.name}")
    os.makedirs(experiment_folder, exist_ok=True)

    # 3. create experiment config
    config_file = os.path.join(experiment_folder, "config.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))

    # 4. generate xpaths
    graph = build_graph()
    seed_dataset = dataset.to_seed()
    xpath_results = defaultdict(list)
    ie_evaluator = IEEvaluator()
    ie_eval_results = []
    def collect_result(id: str, data: dict, ground_truth: list):
        # collect xpaths
        for field_name, field_xpath in data["xpaths"].items():
            xpath_results[id].append(
                {
                    "name": field_name,
                    "xpath": field_xpath,
                }
            )

        # collect ie results
        for snippet, truth in zip(data["snippets"], ground_truth):
            data = {key: snippet["data"][key]["value"] for key in snippet["data"].keys()}
            result = ie_evaluator(data, truth)
            result["id"] = id
            ie_eval_results.append(result)

    for id, seed in tqdm.tqdm(seed_dataset, total=len(seed_dataset), desc="Generating XPaths"):
        output_file = os.path.join(experiment_folder, f"{seed.id}.json")
        if os.path.exists(output_file):
            state = json.loads(open(output_file, "r").read())
            collect_result(id, state, seed.ground_truth)
            continue

        state = {
            "snippets": [
                {
                    "id": md5(html.encode()).hexdigest(),
                    "raw_html": html,
                    "data": {},
                }
                for html in seed.htmls
            ],
            "query": seed.query,
            "xpaths": {},
            "fields": [],
        }
        new_state = graph.invoke(state)

        with open(output_file, "w") as f:
            for snippet in new_state["snippets"]:
                del snippet["raw_html"]
            f.write(json.dumps(new_state, ensure_ascii=False, indent=4))
        collect_result(seed.id, new_state, seed.ground_truth)
    ie_overall = ie_evaluator.to_json()
    ie_overall["id"] = "overall"
    ie_eval_results.append(ie_overall)
    ie_eval_df = pd.DataFrame(ie_eval_results)
    ie_eval_df.to_csv(os.path.join(experiment_folder, "ie_eval_results.csv"), index=False)

    # 5. evaluate xpaths
    xpath_evaluator = XPathEvaluator()
    eval_results = []
    for id, seed in tqdm.tqdm(seed_dataset, total=len(seed_dataset), desc="Evaluating XPaths"):
        samples = dataset[id]
        xpaths = xpath_results[id]
        result = xpath_evaluator(samples, xpaths)
        result["id"] = id
        eval_results.append(result)
    xpath_overall = xpath_evaluator.to_json()
    xpath_overall["id"] = "overall"
    eval_results.append(xpath_overall)
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(os.path.join(experiment_folder, "eval_results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Cmd Line Tool for running experiments")
    parser.add_argument("--dataset", required=True, choices=["SWDE", "SWDE_Extended"], help="Dataset to use")
    parser.add_argument("--data_dir", default="data/swde", help="Data folder")
    parser.add_argument("--ie_model", required=True, help="Information extraction model")
    parser.add_argument("--program_model", required=True, help="Program model")
    parser.add_argument("--ablation", choices=list(ablation_options.keys()), help="Ablation study option")
    parser.add_argument("--eval_sample_size", type=int, default=32, help="Number of samples to evaluate")
    parser.add_argument("--output", default=".tmp", help="Output folder for results, the root folder")
    parser.add_argument("--env", default=".env", help="load environment variables from a env file, default is .env")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("name", help="Experiment Name, Unique Identifier")

    args = parser.parse_args()

    check_model(args.ie_model)
    check_model(args.program_model)

    # 1. Load environment variables
    if args.env:
        load_dotenv(args.env)
        print(f"Loaded environment variables from {args.env}")

    # 2. setup environment variables
    os.environ["PROMPT_LANG"] = "en"
    os.environ["IE_MODEL"] = args.ie_model
    os.environ["PROGRAM_MODEL"] = args.program_model
    os.environ["ABLATION_EXPERIMENT"] = ablation_options.get(args.ablation, "")

    # 3. create dataset
    dataset = create_dataset(
        args.dataset,
        args.data_dir,
        eval_sample_size=args.eval_sample_size,
        seed=args.seed,
    )

    # 4. run experiment
    run_experiment(dataset, args)

if __name__ == "__main__":
    main()
