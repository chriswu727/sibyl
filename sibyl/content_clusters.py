"""Deterministic near-duplicate clustering for retrieved source text."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Sequence, Set


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SHINGLE_SIZE = 5
_MIN_TOKENS = 80
_MIN_UNIQUE_SHINGLES = 20
_MIN_LENGTH_RATIO = 0.55
_MIN_CONTAINMENT = 0.82


@dataclass(frozen=True)
class ContentClusters:
    cluster_ids: List[str]
    cluster_count: int
    duplicate_count: int
    method: str = "token_5gram_containment_v1"


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").casefold())


def _digest(tokens: Sequence[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def _shingles(tokens: Sequence[str]) -> Set[str]:
    if len(tokens) < _SHINGLE_SIZE:
        return set()
    return {
        " ".join(tokens[index:index + _SHINGLE_SIZE])
        for index in range(len(tokens) - _SHINGLE_SIZE + 1)
    }


def _near_duplicate(
    left_tokens: Sequence[str],
    left_shingles: Set[str],
    right_tokens: Sequence[str],
    right_shingles: Set[str],
) -> bool:
    shorter = min(len(left_tokens), len(right_tokens))
    longer = max(len(left_tokens), len(right_tokens))
    if shorter < _MIN_TOKENS or shorter / longer < _MIN_LENGTH_RATIO:
        return False
    if (
        len(left_shingles) < _MIN_UNIQUE_SHINGLES
        or len(right_shingles) < _MIN_UNIQUE_SHINGLES
    ):
        return False
    overlap = len(left_shingles & right_shingles)
    return overlap / min(len(left_shingles), len(right_shingles)) >= _MIN_CONTAINMENT


def cluster_content(texts: Sequence[str]) -> ContentClusters:
    if not texts:
        return ContentClusters([], 0, 0)

    token_groups = [_tokens(text) for text in texts]
    digests = [_digest(tokens) for tokens in token_groups]
    shingle_groups = [_shingles(tokens) for tokens in token_groups]
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            exact_duplicate = (
                bool(token_groups[left]) and digests[left] == digests[right]
            )
            if exact_duplicate or _near_duplicate(
                token_groups[left],
                shingle_groups[left],
                token_groups[right],
                shingle_groups[right],
            ):
                union(left, right)

    members = {}
    for index in range(len(texts)):
        members.setdefault(find(index), []).append(index)
    cluster_id_by_root = {
        root: f"cc_{min(digests[index] for index in indices)[:16]}"
        for root, indices in members.items()
    }
    cluster_ids = [cluster_id_by_root[find(index)] for index in range(len(texts))]
    cluster_count = len(set(cluster_ids))
    return ContentClusters(
        cluster_ids=cluster_ids,
        cluster_count=cluster_count,
        duplicate_count=len(texts) - cluster_count,
    )
