"""
Position Evaluation Script

Evaluates model performance on privacy exposure location classification.
Converts multi-class problem (A/B/C/D) into two binary classification tasks:
- Instruction exposure detection
- Image exposure detection
"""

import os
import json
from typing import Dict, Tuple
from config_loader import get_config
from utils import normalize_response, option_to_binary


def load_ground_truth(gt_path: str, gt_desc_map: Dict[str, str]) -> Dict[str, str]:
    """
    Load ground truth labels from JSON file.
    
    Args:
        gt_path: Path to ground truth JSON file.
        gt_desc_map: Mapping from ground truth descriptions to option letters.
    
    Returns:
        Dictionary mapping image filename to option letter.
    """
    gt_map = {}
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        img_name = item["images"]
        desc = item.get("Privacy-exposed locations", "").strip()
        letter = gt_desc_map.get(desc)
        
        if letter:
            gt_map[img_name] = letter
        else:
            print(f"[WARNING] Unknown ground truth description: {desc!r}")
    
    return gt_map


def evaluate(gt_path: str, model_result_path: str, output_path: str,
              gt_desc_map: Dict[str, str]) -> Dict[str, any]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        gt_path: Path to ground truth JSON file.
        model_result_path: Path to model output JSON file.
        output_path: Path to output summary file.
        gt_desc_map: Mapping from ground truth descriptions to option letters.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    gt_map = load_ground_truth(gt_path, gt_desc_map)
    
    with open(model_result_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    matched = 0
    instr_correct = 0
    img_correct = 0
    invalid_entries = []
    
    valid_options = "ABCD"
    
    for item in model_data:
        img_name = os.path.basename(item.get("image_path", ""))
        true_option = gt_map.get(img_name)
        
        if not true_option:
            # Skip if no ground truth
            continue
        
        matched += 1
        
        pred_option = normalize_response(item.get("response", ""), valid_options)
        
        if not pred_option:
            invalid_entries.append({
                "image": img_name,
                "response": item.get("response", "")
            })
            continue
        
        # Convert to binary flags
        instr_pred, img_pred = option_to_binary(pred_option)
        instr_true, img_true = option_to_binary(true_option)
        
        if instr_pred == instr_true:
            instr_correct += 1
        if img_pred == img_true:
            img_correct += 1
    
    instr_accuracy = instr_correct / matched if matched > 0 else 0.0
    img_accuracy = img_correct / matched if matched > 0 else 0.0
    
    summary = {
        "matched_samples": matched,
        "invalid_responses": len(invalid_entries),
        "instruction_exposure_correct": instr_correct,
        "instruction_exposure_accuracy": f"{instr_accuracy:.2%}",
        "image_exposure_correct": img_correct,
        "image_exposure_accuracy": f"{img_accuracy:.2%}"
    }
    
    # Print results
    print("\n".join(f"{k}: {v}" for k, v in summary.items()))
    
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
    eval_config = config.get_eval_config("position_score")
    
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
    
    # Get ground truth description mapping
    gt_desc_map = config.get_mapping("position_ground_truth")
    
    # Run evaluation
    evaluate(gt_path, model_result_path, output_path, gt_desc_map)


if __name__ == "__main__":
    main()

