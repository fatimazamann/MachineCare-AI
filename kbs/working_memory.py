"""Working memory: the dynamic fact base of one diagnostic session.

Every fact keeps the list of *contributions* that support or oppose it
(user observations, sensor abstractions, asset register, fired
rules).  The fact's certainty is the MYCIN combination of all contributions,
which makes the result independent of the order in which evidence arrived and
lets the system detect and explain conflicting evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from . import certainty as cfm


@dataclass
class Contribution:
    source_id: str            # unique id, e.g. 'operator', 'sensor:high_vibration:0', 'R-CAV-1'
    source_type: str          # operator | technician | sensor | asset | default | rule
    cf: float
    detail: str = ""
    rule_cf: Optional[float] = None
    premise_cf: Optional[float] = None
    cycle: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class WorkingMemory:
    def __init__(self):
        self._facts: Dict[str, Dict[str, Contribution]] = {}
        self.log: List[dict] = []

    # -------------------------------------------------------------- update
    def put(self, fact: str, contribution: Contribution) -> bool:
        """Insert/replace the contribution of one source; True if CF changed."""
        before = self.cf(fact)
        self._facts.setdefault(fact, {})[contribution.source_id] = contribution
        after = self.cf(fact)
        self.log.append({"op": "put", "fact": fact, "source": contribution.source_id,
                         "cf": contribution.cf, "before": before, "after": after})
        return abs(after - before) > 1e-9

    def retract(self, fact: str, source_id: str) -> bool:
        if fact in self._facts and source_id in self._facts[fact]:
            before = self.cf(fact)
            del self._facts[fact][source_id]
            if not self._facts[fact]:
                del self._facts[fact]
            self.log.append({"op": "retract", "fact": fact, "source": source_id,
                             "before": before, "after": self.cf(fact)})
            return True
        return False

    # -------------------------------------------------------------- query
    def cf(self, fact: str) -> float:
        contribs = self._facts.get(fact)
        if not contribs:
            return 0.0
        return cfm.combine_all(c.cf for c in contribs.values())

    def known(self, fact: str) -> bool:
        return fact in self._facts

    def contributions(self, fact: str) -> List[Contribution]:
        return list(self._facts.get(fact, {}).values())

    def contribution(self, fact: str, source_id: str) -> Optional[Contribution]:
        return self._facts.get(fact, {}).get(source_id)

    def facts(self, prefix: Optional[str] = None) -> Dict[str, float]:
        out = {}
        for f in self._facts:
            if prefix is None or f.startswith(prefix):
                out[f] = self.cf(f)
        return out

    def evidence_facts(self) -> Dict[str, float]:
        return {f: self.cf(f) for f in self._facts if ":" not in f}

    def combination_steps(self, fact: str) -> List[dict]:
        """Step-by-step MYCIN combination for the explanation facility."""
        steps, acc = [], 0.0
        for c in self._facts.get(fact, {}).values():
            new = cfm.combine(acc, c.cf)
            steps.append({"source": c.source_id, "cf": round(c.cf, 4), "before": round(acc, 4), "after": round(new, 4)})
            acc = new
        return steps

    # -------------------------------------------------------------- conflicts
    def conflicts(self, threshold: float = 0.3) -> List[dict]:
        """Facts supported AND opposed by independent sources."""
        out = []
        for fact, contribs in self._facts.items():
            pos = [c for c in contribs.values() if c.cf >= threshold]
            neg = [c for c in contribs.values() if c.cf <= -threshold]
            if pos and neg:
                dominant = max(contribs.values(), key=lambda c: abs(c.cf))
                out.append({
                    "fact": fact,
                    "for": [c.to_dict() for c in pos],
                    "against": [c.to_dict() for c in neg],
                    "resolved_cf": round(self.cf(fact), 4),
                    "dominant_source": dominant.source_id,
                    "dominant_type": dominant.source_type,
                })
        return out
