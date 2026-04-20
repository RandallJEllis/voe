"""Tests for Gray code utilities."""

import pytest
from regsens.gray_code import gray, gray_diff, gray_to_mask, gray_active_positions, popcount


def test_gray_k3_known_sequence():
    """G(0..7) == [0, 1, 3, 2, 6, 7, 5, 4]."""
    expected = [0, 1, 3, 2, 6, 7, 5, 4]
    assert [gray(i) for i in range(8)] == expected


@pytest.mark.parametrize("k", [4, 8, 12])
def test_consecutive_differ_one_bit(k):
    """Any consecutive Gray codes differ in exactly one bit."""
    for i in range(1, 1 << k):
        diff = gray(i) ^ gray(i - 1)
        assert diff & (diff - 1) == 0, f"Multiple bits changed at i={i}"


@pytest.mark.parametrize("k", [4, 8])
def test_gray_diff_bit_position_correct(k):
    for i in range(1, 1 << k):
        bit_pos, is_add = gray_diff(i)
        diff = gray(i) ^ gray(i - 1)
        assert diff == (1 << bit_pos), f"Wrong bit_pos at i={i}"


@pytest.mark.parametrize("k", [4, 8])
def test_gray_diff_direction_correct(k):
    for i in range(1, 1 << k):
        bit_pos, is_add = gray_diff(i)
        expected_add = bool((gray(i) >> bit_pos) & 1)
        assert is_add == expected_add, f"Wrong is_add at i={i}"


@pytest.mark.parametrize("k", [3, 5, 8])
def test_gray_to_mask_roundtrip(k):
    for g in range(1 << k):
        mask = gray_to_mask(g, k)
        assert len(mask) == k
        reconstructed = sum(int(v) << j for j, v in enumerate(mask))
        assert reconstructed == g


@pytest.mark.parametrize("k", [4, 6])
def test_gray_covers_all_subsets(k):
    """Gray code visits all 2^k distinct subsets."""
    seen = set(gray(i) for i in range(1 << k))
    assert len(seen) == 1 << k


def test_popcount():
    assert popcount(0) == 0
    assert popcount(1) == 1
    assert popcount(7) == 3    # 111
    assert popcount(255) == 8
