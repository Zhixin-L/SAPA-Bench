"""
Action and Coordinate Evaluation Script

Evaluates model performance on action type prediction and coordinate localization.
Scoring scheme:
- Action match: 50 points
- Coordinate match: 50 points (based on distance)
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from config_loader import get_config
from utils import parse_coordinates


def score_one(gt_action: str, gt_coord: Optional[Tuple[int, int]],
              pred_action: Optional[str], pred_coord: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Score a single prediction against ground truth.
    
    Args:
        gt_action: Ground truth action type.
        gt_coord: Ground truth coordinates (x, y) or None.
        pred_action: Predicted action type or None.
        pred_coord: Predicted coordinates (x, y) or None.
    
    Returns:
        Tuple of (action_score, coord_score) where each is 0-50.
    """
    # Action matching (50 points)
    if gt_action.upper() == "TEXT":
        match_type = pred_action and pred_action.upper() in ("TEXT", "TYPE")
    else:
        match_type = pred_action and pred_action.upper() == gt_action.upper()
    
    action_score = 50 if match_type else 0
    
    # If action is wrong, coordinate score is 0
    if not match_type:
        return action_score, 0
    
    # Coordinate matching (50 points)
    # If GT is TEXT, no coordinates needed - give full score
    if gt_action.upper() == "TEXT":
        return action_score, 50
    
    # If GT requires coordinates, both must be present
    if gt_coord is None or pred_coord is None:
        return action_score, 0
    
    # Calculate distance and linear decay
    gx, gy = gt_coord
    px, py = pred_coord
    distance = math.hypot(px - gx, py - gy)
    max_distance = math.hypot(1000, 1000)
    coord_score = round(50 * max(0.0, 1 - distance / max_distance))
    
    return action_score, coord_score


def main():
    """Main evaluation function."""
    config = get_config()
    
    # Get paths
    eval_config = config.get_eval_config("acc_score")
    
    pred_file = eval_config["pred_file"]
    pred_path = config.resolve_path(
        config.get_path("model_results_dir"),
        pred_file
    )
    
    # Check if sampled_data is configured
    try:
        sampled_data_path = config.get_path("sampled_data")
        if not sampled_data_path or sampled_data_path.strip() == "":
            raise KeyError("sampled_data not configured")
        gt_path = config.resolve_path(sampled_data_path)
    except KeyError:
        print("Error: Action evaluation requires 'sampled_data' to be configured in config.json")
        print("Please add 'sampled_data' to the 'paths' section in your config.json file.")
        print("Example: \"sampled_data\": \"data/sampled_1100.json\"")
        return
    
    output_file = eval_config["output_file"]
    output_path = config.resolve_path(
        config.get_path("model_results_dir"),
        output_file
    )
    
    # Load data
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # Create index for predictions
    pred_index = {Path(r["image_path"]).name: r for r in predictions}
    
    results = []
    
    for gt in ground_truths:
        img_name = gt["screenshot"]
        gt_action = gt["action"]
        gt_coord = parse_coordinates(gt.get("ps", ""))
        
        record = {
            "screenshot": img_name,
            "gt_action": gt_action,
            "gt_coord": gt_coord
        }
        
        pred = pred_index.get(img_name)
        
        if not pred:
            record.update({
                "parsed_action": None,
                "parsed_coord": None,
                "action_score": 0,
                "coord_score": 0,
                "total_score": 0
            })
        else:
            pred_action = pred.get("parsed_action")
            pred_coord = pred.get("parsed_coord")
            
            action_score, coord_score = score_one(
                gt_action, gt_coord, pred_action, pred_coord
            )
            
            record.update({
                "parsed_action": pred_action,
                "parsed_coord": pred_coord,
                "action_score": action_score,
                "coord_score": coord_score,
                "total_score": action_score + coord_score
            })
        
        results.append(record)
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation complete. Saved {len(results)} scored records to {output_path}")


if __name__ == "__main__":
    main()

