# SAPA-Bench Evaluation Suite

Evaluation scripts for the SAPA-Bench privacy awareness benchmark. This suite provides comprehensive evaluation metrics for assessing model performance on privacy-related tasks.

## Overview

This evaluation suite includes the following evaluation scripts:

1. **Security Level Evaluation** (`evaluate_level.py`) - Evaluates model performance on security level classification (Low/Mid/High/No private content)
2. **Existence Evaluation** (`evaluate_exist.py`) - Evaluates the proportion of predictions labeled as option 'A'
3. **Category Evaluation** (`evaluate_category.py`) - Evaluates model performance on privacy category classification (9 categories)
4. **Position Evaluation** (`evaluate_position.py`) - Evaluates privacy exposure location detection (instruction/image/both/none)
5. **Action Evaluation** (`evaluate_action.py`) - Evaluates action type prediction and coordinate localization (requires `sampled_data` in config)
6. **Rationale and Action Evaluation** (`evaluate_ra.py`) - Uses GPT-4o to automatically evaluate model responses

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `config.json` and update the paths according to your setup:

```json
{
  "paths": {
    "ground_truth": "data/label_6.22.json",
    "model_results_dir": "data/results",
    "output_dir": "output/scores"
  },
  ...
}
```

**Optional Configuration:** If you plan to run action and coordinate evaluation (`evaluate_action.py`), you can optionally add `"sampled_data": "data/sampled_1100.json"` to the `paths` section. This file contains ground truth data for action types and coordinates. If this field is not configured, `evaluate_action.py` will display an error message with instructions.

2. For RA evaluation scripts, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or update `config.json`:

```json
{
  "api": {
    "api_key": "your-api-key-here",
    ...
  }
}
```

## Usage

### Basic Evaluation Scripts

Run individual evaluation scripts:

```bash
# Security level evaluation
python evaluate_level.py

# Category evaluation
python evaluate_category.py

# Position evaluation
python evaluate_position.py

# Action and coordinate evaluation
python evaluate_action.py

# Existence evaluation
python evaluate_exist.py
```

### RA Evaluation Scripts

The RA evaluation scripts use GPT-4o to automatically score model responses:

```bash
# Standard RA evaluation
python evaluate_ra.py

```

These scripts support resume capability - if interrupted, they will continue from where they left off.



### Ground Truth Format

The ground truth JSON file should contain entries with the following structure:

```json
[
  {
    "images": "image_filename.jpg",
    "conversations1": {
      "security_type": "Low"
    },
    "image_type": "Location and environmental data leakage",
    "Privacy-exposed locations": "Private content in the instructions"
  },
  ...
]
```

### Model Output Format

Model output JSON files should contain entries with the following structure:

```json
[
  {
    "image_path": "/path/to/image_filename.jpg",
    "response": "A"
  },
  ...
]
```

For action evaluation, the format should be:

```json
[
  {
    "image_path": "/path/to/screenshot.jpg",
    "parsed_action": "CLICK",
    "parsed_coord": [100, 200]
  },
  ...
]
```

## Output

Evaluation results are saved to the output directory specified in `config.json`. Summary statistics are printed to the console and saved to JSON files.

## Environment Variables

You can override configuration using environment variables:

- `OPENAI_API_KEY`: OpenAI API key for RA evaluation
- `OPENAI_BASE_URL`: Base URL for OpenAI API (default: https://api.openai.com/v1)
- `GROUND_TRUTH_PATH`: Path to ground truth file
- `OUTPUT_DIR`: Output directory for results

## Project Structure

```
.
├── config.json              # Configuration file
├── config_loader.py          # Configuration loading utilities
├── utils.py                  # Common utility functions
├── evaluate_level.py         # Security level evaluation
├── evaluate_exist.py         # Existence evaluation
├── evaluate_category.py      # Category evaluation
├── evaluate_position.py      # Position evaluation
├── evaluate_action.py        # Action and coordinate evaluation
├── evaluate_ra.py            # RA evaluation (base)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```



## Citation


```bibtex

@inproceedings{lin2025sapa,
  title      = {Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents},
  author     = {Lin, Zhixin and Li, Jungang and Pan, Shidong and Shi, Yibo and Yao, Yue and Xu, Dongliang},
  booktitle  = {The Fortieth Annual AAAI Conference on Artificial Intelligence},
  year       = {2026},
}
```


