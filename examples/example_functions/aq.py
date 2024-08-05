"""Simple function to calculate the Alphanumeric Quabala (AQ) value of a string."""

import re
from typing import List, Tuple


__all__ = ["aq", "aq_multiple"]


def aq(input_string: str) -> int:
    """
    Calculate the Alphanumeric Quabala (AQ) value of a string.

    This function calculates the sum of the numeric values of digits
    and the positional values of letters in the input string.

    Args:
        input_string (str): The input string to calculate the AQ for.

    Returns:
        int: The calculated AQ value.
    """
    input_string = re.sub(r"[^a-zA-Z0-9]", "", input_string.lower())
    digits = sum(int(char) for char in input_string if char.isdigit())
    letters = sum(ord(char) - 96 for char in input_string if char.isalpha())
    return digits + letters



def aq_multiple(input_strings: List[str]) -> List[Tuple[str, int]]:
    """
    Calculate the Alphanumeric Quabala (AQ) value for multiple strings.

    This function calculates the AQ value for each string in the input list
    and returns a sorted list of arrays containing the original string and its AQ value
    """
    return sorted([(s, aq(s)) for s in input_strings], key=lambda x: x[1])
