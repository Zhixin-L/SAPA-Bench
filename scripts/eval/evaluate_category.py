"""
Category Evaluation Script

Evaluates model performance on privacy category classification task.
Maps model responses (A-I) to privacy category labels.
"""

import os
import json
from typing import Dict
from config_loader import get_config
from utils import normalize_response


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """
    Load ground truth labels from JSON file.
    
    Args:
        gt_path: Path to ground truth JSON file.
    
    Returns:
        Dictionary mapping image filename to image type.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {item["images"]: item["image_type"] for item in data}


def evaluate(gt_path: str, model_result_path: str, output_path: str,
              option_map: Dict[str, str]) -> Dict[str, any]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        gt_path: Path to ground truth JSON file.
        model_result_path: Path to model output JSON file.
        output_path: Path to output summary file.
        option_map: Mapping from option letters to category labels.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    gt_map = load_ground_truth(gt_path)
    
    with open(model_result_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    matched = 0
    correct = 0
    invalid_entries = []
    
    valid_options = ''.join(option_map.keys())
    
    for item in model_data:
        filename = os.path.basename(item.get("image_path", ""))
        
        if filename not in gt_map:
            # Skip if no ground truth
            continue
        
        matched += 1
        
        response = item.get("response", "")
        pred_option = normalize_response(response, valid_options)
        
        if not pred_option:
            invalid_entries.append({"image": filename, "response": response})
            continue
        
        if option_map[pred_option] == gt_map[filename]:
            correct += 1
    
    accuracy = correct / matched if matched > 0 else 0.0
    
    summary = {
        "matched_samples": matched,
        "invalid_responses": len(invalid_entries),
        "correct_predictions": correct,
        "accuracy": f"{accuracy:.2%}"
    }
    
    # Print results
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
        f.write("\n")
    
    print(f"\nEvaluation results appended to {output_path}")
    
    return summary


def main():
    """Main evaluation function."""
    config = get_config()
    
    # Get paths
    gt_path = config.resolve_path(config.get_path("ground_truth"))
    eval_config = config.get_eval_config("cat_score")
    
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
    option_map = config.get_mapping("category_options")
    
    # Run evaluation
    evaluate(gt_path, model_result_path, output_path, option_map)


if __name__ == "__main__":
    main()

