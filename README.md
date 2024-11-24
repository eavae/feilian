# XPath Agent

XPath Agent is an advanced language model-based crawler designed to efficiently extract information from massive html files.

![workflow of XPath Agent](paper/workflow.png)

## How to Reproduce

To reproduce the results presented in the associated paper, follow the steps below using the provided command-line interface script.

### Prerequisites

Ensure you have the necessary dependencies installed and your environment is properly configured.

For example, if you are using OpenAI API, ensure following environment variables are setting correctly:

```bash
OPENAI_API_KEY=sk-xxx
```

### Running the Experiment

Use the following command to run the experiment:

```bash
python scripts/experiment_cli.py --dataset <DATASET> --ie_model <IE_MODEL> --program_model <PROGRAM_MODEL> --name <EXPERIMENT_NAME>
```

### Command-Line Arguments

- `--dataset`: Specifies the dataset to use. Options are `SWDE` or `SWDE_Extended`.
- `--data_dir`: The directory where the dataset is stored. Default is `data/swde`.
- `--ie_model`: The information extraction model to be used.
- `--program_model`: The program model to be used.
- `--ablation`: Optional. Specifies the ablation study option. Choices are 
- `--eval_sample_size`: The number of samples to evaluate. Default is `32`.
- `--output`: The directory where the results will be saved. Default is `.tmp`.
- `--env`: The environment file to load variables from. Default is `.env`.
- `--seed`: The random seed for reproducibility. Default is `0`.
- `name`: A unique identifier for the experiment.
