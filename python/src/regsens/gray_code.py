"""
Gray code enumeration utilities.

The n-bit reflected Gray code visits all 2^n subsets such that consecutive
subsets differ by exactly one element. This maps perfectly to O(mp) QR updates:
each step either adds or removes exactly one column.
"""


def gray(i: int) -> int:
    """Standard reflected Gray code: G(i) = i XOR (i >> 1)."""
    return i ^ (i >> 1)


def gray_diff(i: int) -> tuple[int, bool]:
    """
    Return (bit_pos, is_add) for the single bit that changes from G(i-1) to G(i).

    bit_pos = position of the lowest set bit of i  (= count_trailing_zeros(i))
    is_add  = True  if that bit is 1 in G(i)  → column is ADDED
              False if that bit is 0 in G(i)  → column is REMOVED

    Precondition: i >= 1.
    """
    lsb = i & (-i)
    bit_pos = lsb.bit_length() - 1
    is_add = bool((gray(i) >> bit_pos) & 1)
    return bit_pos, is_add


def gray_to_mask(g: int, k: int) -> list[bool]:
    """Convert Gray code integer g to a boolean inclusion mask of length k."""
    return [bool((g >> j) & 1) for j in range(k)]


def gray_active_positions(g: int, k: int) -> list[int]:
    """Return sorted 0-based positions of set bits in Gray code value g."""
    return [j for j in range(k) if (g >> j) & 1]


def popcount(g: int) -> int:
    """Count the number of set bits in g."""
    return bin(g).count("1")
