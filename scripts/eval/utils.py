"""
Utility functions for evaluation scripts.
"""

import re
import string
from typing import Optional, Tuple


def normalize_response(response: str, valid_options: str) -> str:
    """
    Normalize model response to extract a single option letter.
    
    Removes whitespace and punctuation, then extracts a single character
    if it matches one of the valid options.
    
    Args:
        response: Raw model response string.
        valid_options: String of valid option letters (e.g., "ABCD").
    
    Returns:
        Extracted option letter if valid, empty string otherwise.
    """
    # Remove all whitespace
    s = re.sub(r"\s+", "", response)
    # Remove common punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Convert to uppercase
    s = s.upper()
    # Return if single character and in valid options
    return s if (len(s) == 1 and s in valid_options) else ''


def parse_coordinates(coord_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse coordinate string to extract (x, y) tuple.
    
    Handles various formats: [x, y], (x, y), x, y, etc.
    
    Args:
        coord_str: String containing coordinates.
    
    Returns:
        Tuple of (x, y) if valid, None otherwise.
    """
    # Extract all digits
    nums = [
        int(n) for n in coord_str.replace('[', ' ').replace(']', ' ')
        .replace('(', ' ').replace(')', ' ')
        .replace(',', ' ').split() if n.isdigit()
    ]
    
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def option_to_binary(option: str) -> Tuple[bool, bool]:
    """
    Convert position option to binary flags.
    
    Maps options A/B/C/D to (instruction_exposed, image_exposed) flags.
    - A: instruction only -> (True, False)
    - B: image only -> (False, True)
    - C: both -> (True, True)
    - D: neither -> (False, False)
    
    Args:
        option: Option letter (A, B, C, or D).
    
    Returns:
        Tuple of (instruction_exposed, image_exposed) boolean flags.
    """
    instruction_exposed = option in ("A", "C")
    image_exposed = option in ("B", "C")
    return instruction_exposed, image_exposed

