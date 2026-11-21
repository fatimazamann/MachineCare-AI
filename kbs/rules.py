"""Production rules and the parser for the rule language used in rules.kb."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import certainty as cfm

# Conflict resolution priority: safety rules first, then diagnostic rules, then root-cause rules
SALIENCE = {"safety": 100, "diagnostic": 50, "cause": 30}
RULE_TYPES = set(SALIENCE)


class RuleSyntaxError(ValueError):
    pass


@dataclass
class Condition:
    """One premise of a rule.

    kind = 'fact'  -> CF(fact)
    kind = 'not'   -> -CF(fact)
    kind = 'any'   -> max CF over the listed items (items may be 'NOT x')
    """

    kind: str
    facts: List[str]
    negated_items: List[bool] = field(default_factory=list)

    def evaluate(self, get_cf: Callable[[str], float]) -> float:
        if self.kind == "fact":
            return get_cf(self.facts[0])
        if self.kind == "not":
            return cfm.negation(get_cf(self.facts[0]))
        values = []
        for fact, neg in zip(self.facts, self.negated_items):
            v = get_cf(fact)
            values.append(cfm.negation(v) if neg else v)
        return cfm.disjunction(values)

    def referenced(self) -> List[str]:
        return list(self.facts)

    def text(self) -> str:
        if self.kind == "fact":
            return self.facts[0]
        if self.kind == "not":
            return "NOT " + self.facts[0]
        parts = [("NOT " if n else "") + f for f, n in zip(self.facts, self.negated_items)]
        return "ANY(" + ", ".join(parts) + ")"


@dataclass
class Rule:
    id: str
    type: str
    domain: List[str]
    conditions: List[Condition]
    conclusion: str
    cf: float
    sources: List[str] = field(default_factory=list)
    note: str = ""
    salience: int = 50
    line: int = 0

    @property
    def specificity(self) -> int:
        return len(self.conditions)

    def premise_facts(self) -> List[str]:
        out: List[str] = []
        for c in self.conditions:
            for f in c.referenced():
                if f not in out:
                    out.append(f)
        return out

    def premise_cf(self, get_cf: Callable[[str], float]) -> float:
        return cfm.conjunction(c.evaluate(get_cf) for c in self.conditions)

    def text(self) -> str:
        cond = " AND ".join(c.text() for c in self.conditions)
        return f"IF {cond} THEN {self.conclusion} (CF {self.cf:+.2f})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "domain": self.domain,
            "if": [c.text() for c in self.conditions],
            "then": self.conclusion,
            "cf": self.cf,
            "salience": self.salience,
            "sources": self.sources,
            "note": self.note,
            "text": self.text(),
        }


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")
_FACT_RE = re.compile(r"^(?:(?:fault|cause|safety):)?[a-z][a-z0-9_]*$")


def _split_top_level(text: str, sep: str) -> List[str]:
    """Split on a keyword separator that is not inside parentheses."""
    parts, depth, buf, i = [], 0, "", 0
    pattern = f" {sep} "
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and text[i:i + len(pattern)].upper() == pattern:
            parts.append(buf.strip())
            buf = ""
            i += len(pattern)
            continue
        buf += ch
        i += 1
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _parse_item(token: str, line_no: int):
    token = token.strip()
    neg = False
    if token.upper().startswith("NOT "):
        neg = True
        token = token[4:].strip()
    if not _FACT_RE.match(token):
        raise RuleSyntaxError(f"line {line_no}: invalid fact name '{token}'")
    return token, neg


def parse_condition(text: str, line_no: int) -> Condition:
    text = text.strip()
    if text.upper().startswith("ANY(") and text.endswith(")"):
        inner = text[4:-1]
        items = [p for p in (x.strip() for x in inner.split(",")) if p]
        if len(items) < 2:
            raise RuleSyntaxError(f"line {line_no}: ANY() needs at least two items")
        parsed = [_parse_item(p, line_no) for p in items]
        return Condition("any", [p[0] for p in parsed], [p[1] for p in parsed])
    fact, neg = _parse_item(text, line_no)
    return Condition("not" if neg else "fact", [fact], [neg])


def parse_rules(text: str) -> List[Rule]:
    rules: List[Rule] = []
    current: Optional[dict] = None
    in_if = False

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip() if not raw.strip().startswith("NOTE") else raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        keyword, _, rest = stripped.partition(" ")
        kw = keyword.upper()
        rest = rest.strip()

        if kw == "RULE":
            if current is not None:
                raise RuleSyntaxError(f"line {line_no}: RULE inside rule '{current['id']}' (missing END)")
            if not _ID_RE.match(rest):
                raise RuleSyntaxError(f"line {line_no}: invalid rule id '{rest}'")
            current = {"id": rest, "line": line_no, "conditions": [], "sources": [], "note": "",
                       "domain": [], "type": None, "then": None, "cf": None, "salience": None}
            in_if = False
            continue

        if current is None:
            raise RuleSyntaxError(f"line {line_no}: statement outside a RULE block: '{stripped}'")

        if kw == "TYPE":
            t = rest.lower()
            if t not in RULE_TYPES:
                raise RuleSyntaxError(f"line {line_no}: unknown rule type '{rest}'")
            current["type"] = t
            in_if = False
        elif kw == "DOMAIN":
            current["domain"] = [d.strip() for d in rest.split(",") if d.strip()]
            in_if = False
        elif kw == "IF":
            for part in _split_top_level(rest, "AND"):
                current["conditions"].append(parse_condition(part, line_no))
            in_if = True
        elif kw == "AND":
            if not in_if:
                raise RuleSyntaxError(f"line {line_no}: AND continuation outside IF")
            for part in _split_top_level(rest, "AND"):
                current["conditions"].append(parse_condition(part, line_no))
        elif kw == "THEN":
            m = re.match(r"^(\S+)\s+CF\s+([-+]?\d*\.?\d+)$", rest, flags=re.I)
            if not m:
                raise RuleSyntaxError(f"line {line_no}: THEN must be '<fact> CF <number>'")
            concl, cf = m.group(1), float(m.group(2))
            if not _FACT_RE.match(concl) or ":" not in concl:
                raise RuleSyntaxError(f"line {line_no}: conclusion must be fault:/cause:/safety:<id>")
            if not -1.0 <= cf <= 1.0:
                raise RuleSyntaxError(f"line {line_no}: CF must lie in [-1, 1]")
            current["then"], current["cf"] = concl, cf
            in_if = False
        elif kw == "SOURCE":
            current["sources"] = [s.strip() for s in rest.split(",") if s.strip()]
            in_if = False
        elif kw == "NOTE":
            current["note"] = rest
            in_if = False
        elif kw == "SALIENCE":
            current["salience"] = int(rest)
            in_if = False
        elif kw == "END":
            for key in ("type", "then"):
                if current[key] is None:
                    raise RuleSyntaxError(f"rule {current['id']} (line {current['line']}): missing {key.upper()}")
            if not current["conditions"]:
                raise RuleSyntaxError(f"rule {current['id']}: missing IF")
            if not current["domain"]:
                raise RuleSyntaxError(f"rule {current['id']}: missing DOMAIN")
            rules.append(Rule(
                id=current["id"], type=current["type"], domain=current["domain"],
                conditions=current["conditions"], conclusion=current["then"], cf=current["cf"],
                sources=current["sources"], note=current["note"],
                salience=current["salience"] if current["salience"] is not None else SALIENCE[current["type"]],
                line=current["line"],
            ))
            current = None
            in_if = False
        else:
            raise RuleSyntaxError(f"line {line_no}: unknown keyword '{keyword}'")

    if current is not None:
        raise RuleSyntaxError(f"rule {current['id']}: missing END")
    ids = [r.id for r in rules]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise RuleSyntaxError(f"duplicate rule ids: {sorted(dup)}")
    return rules
