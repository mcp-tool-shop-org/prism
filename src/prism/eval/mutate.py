"""AST mutation operators that turn a test-PASSING program into a candidate buggy one.

A clean (test-passing) family generation is mutated at ONE site to plant a defect; the mutant is
then RE-RUN against the hidden tests (in ``familygen``) and kept ONLY if it now fails — a
still-passing mutant is "equivalent" (no behavioral change) and is discarded, never mislabeled
buggy. This yields execution-VERIFIED buggy labels that still carry the producing family's style (a
single-operator edit preserves the family signature self-recognition needs to fire — Panickssery
et al. 2024, arXiv:2404.13076).

Operators are the ones Just et al. 2014 (DOI:10.1145/2635868.2635929) found well-COUPLED to real
faults — relational- and arithmetic-operator replacement (ROR / AOR) and logical-connector
replacement (LCR). Constant replacement is deliberately EXCLUDED: that study found it the most
poorly coupled operator (it tends to produce trivially-detectable or equivalent mutants).
Statement-deletion (also well-coupled) is a documented future operator; for v1 the "missing
statement" defect class is covered organically by the natural failed generations ``familygen`` keeps
as buggy.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

# Relational-operator replacement: each comparison op maps to its boundary/negation neighbor. These
# flips change which side of a boundary a branch takes — the classic off-by-one / wrong-comparison.
_ROR: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

# Arithmetic-operator replacement: swap an arithmetic op for a related one that changes the value.
_AOR: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
    ast.Pow: ast.Mult,
}

# Logical-connector replacement: and <-> or.
_LCR: dict[type[ast.boolop], type[ast.boolop]] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


@dataclass(frozen=True)
class Mutant:
    """One single-site mutation of a source program (not yet execution-verified)."""

    code: str
    operator: str  # "ror" | "aor" | "lcr"
    description: str  # e.g. "Lt->LtE"


def _collect_sites(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Return ``(walk_index, operator_kind, description)`` for every mutable op, in walk order.

    ``walk_index`` is the position of the owning node in ``ast.walk(tree)``; because a structural
    deep-copy yields an identical walk order, that index locates the same node in a copy without
    relying on node identity or mutating the tree with bookkeeping attributes.
    """
    sites: list[tuple[int, str, str]] = []
    for idx, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare):
            first = type(node.ops[0])
            if first in _ROR:
                sites.append((idx, "ror", f"{first.__name__}->{_ROR[first].__name__}"))
        elif isinstance(node, ast.BinOp):
            akind = type(node.op)
            if akind in _AOR:
                sites.append((idx, "aor", f"{akind.__name__}->{_AOR[akind].__name__}"))
        elif isinstance(node, ast.BoolOp):
            bkind = type(node.op)
            if bkind in _LCR:
                sites.append((idx, "lcr", f"{bkind.__name__}->{_LCR[bkind].__name__}"))
    return sites


def _apply(node: ast.AST, operator: str) -> None:
    """Mutate ``node``'s operator in place (only the first op of a chained comparison flips)."""
    if operator == "ror" and isinstance(node, ast.Compare):
        node.ops[0] = _ROR[type(node.ops[0])]()
    elif operator == "aor" and isinstance(node, ast.BinOp):
        node.op = _AOR[type(node.op)]()
    elif operator == "lcr" and isinstance(node, ast.BoolOp):
        node.op = _LCR[type(node.op)]()


def generate_mutants(code: str, *, limit: int = 8) -> list[Mutant]:
    """Generate up to ``limit`` single-site operator-replacement mutants of ``code``.

    Deterministic (walk order). Returns ``[]`` for un-parseable code or code with no mutable op.
    No-op mutants (those that unparse identically to the source) and duplicates are dropped; callers
    are expected to execution-verify each mutant and keep only those that now fail the tests.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    base = ast.unparse(tree)
    out: list[Mutant] = []
    seen: set[str] = set()
    for walk_index, operator, description in _collect_sites(tree):
        new_tree = copy.deepcopy(tree)
        target = list(ast.walk(new_tree))[walk_index]
        _apply(target, operator)
        try:
            mutant_code = ast.unparse(new_tree)
        except Exception:  # an un-unparseable mutant is skipped, never raised
            continue
        if mutant_code == base or mutant_code in seen:
            continue
        seen.add(mutant_code)
        out.append(Mutant(code=mutant_code, operator=operator, description=description))
        if len(out) >= limit:
            break
    return out
