"""
Existence Evaluation Script

Evaluates the proportion of predictions labeled as option 'A'.
"""

import os
import json
from typing import Dict
from config_loader import get_config


def evaluate(gt_path: str, model_result_path: str, output_path: str) -> Dict[str, any]:
    """
    Evaluate the proportion of 'A' predictions.
    
    Args:
        gt_path: Path to ground truth JSON file.
        model_result_path: Path to model output JSON file.
        output_path: Path to output summary file.
    
    Returns:
        Dictionary containing evaluation statistics.
    """
    # Load model results
    with open(model_result_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Load ground truth to get image set
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # Build set of images in ground truth
    gt_images = {item['images'] for item in gt_data}
    
    # Count predictions
    total = 0
    count_A = 0
    
    for entry in model_data:
        filename = os.path.basename(entry['image_path'])
        if filename in gt_images:
            total += 1
            if entry.get('response') == 'A':
                count_A += 1
    
    # Calculate proportion
    proportion = count_A / total if total > 0 else 0.0
    
    results = {
        "total_matched_entries": total,
        "count_A_predictions": count_A,
        "proportion_A": f"{proportion:.2%}"
    }
    
    # Print results
    print(f"Total matched entries: {total}")
    print(f"Count of 'A' predictions: {count_A}")
    print(f"Proportion of 'A': {proportion:.2%}")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    """Main evaluation function."""
    config = get_config()
    
    # Get paths
    gt_path = config.resolve_path(config.get_path("ground_truth"))
    eval_config = config.get_eval_config("exist_score")
    
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
    
    # Run evaluation
    evaluate(gt_path, model_result_path, output_path)


if __name__ == "__main__":
    main()

