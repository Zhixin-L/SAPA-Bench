"""
Security Level Evaluation Script

Evaluates model performance on security level classification task.
Maps model responses (A/B/C/D) to security levels (Low/Mid/High/No private content).
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple
from config_loader import get_config
from utils import normalize_response


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """
    Load ground truth labels from JSON file.
    
    Args:
        gt_path: Path to ground truth JSON file.
    
    Returns:
        Dictionary mapping image filename to security type.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_map = {}
    for item in data:
        filename = item["images"]
        security_type = item["conversations1"]["security_type"]
        gt_map[filename] = security_type
    
    return gt_map


def evaluate(gt_path: str, model_result_path: str, output_path: str, 
              option_map: Dict[str, str]) -> Dict[str, any]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        gt_path: Path to ground truth JSON file.
        model_result_path: Path to model output JSON file.
        output_path: Path to output summary file.
        option_map: Mapping from option letters to security level labels.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    gt_map = load_ground_truth(gt_path)
    
    with open(model_result_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    matched = 0      # Number of samples with ground truth
    correct = 0      # Number of correct predictions
    invalid = 0      # Number of invalid responses
    
    valid_options = ''.join(option_map.keys())
    
    for item in model_data:
        filename = os.path.basename(item["image_path"])
        true_label = gt_map.get(filename)
        
        if true_label is None:
            # Skip if no ground truth available
            continue
        
        matched += 1
        
        response = item.get("response", "")
        pred_option = normalize_response(response, valid_options)
        
        if not pred_option:
            invalid += 1
            continue
        
        pred_label = option_map[pred_option]
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / matched if matched > 0 else 0.0
    
    summary = {
        "matched_samples": matched,
        "invalid_responses": invalid,
        "correct_predictions": correct,
        "accuracy": f"{accuracy:.2%}"
    }
    
    # Save summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
        f.write("\n")
    
    # Print results
    print(f"Total matched samples: {matched}")
    print(f"Invalid responses: {invalid}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy (correct/total): {accuracy:.2%}")
    
    return summary


def main():
    """Main evaluation function."""
    config = get_config()
    
    # Get paths
    gt_path = config.resolve_path(config.get_path("ground_truth"))
    eval_config = config.get_eval_config("level_score")
    
    model_result_file = eval_config["model_result_file"]
    model_result_path = config.resolve_path(
        config.get_path("model_results_dir"), 
        model_result_file
    )
    
    output_file = eval_config["output_summary"]
    output_path = config.resolve_path(
        config.get_path("output_dir"),
        output_file
    )
    
    # Get option mapping
    option_map = config.get_mapping("level_options")
    
    # Run evaluation
    evaluate(gt_path, model_result_path, output_path, option_map)


if __name__ == "__main__":
    main()

