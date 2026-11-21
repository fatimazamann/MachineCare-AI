"""Knowledge base: loads the frame ontology, evidence vocabulary, fault/cause/
action frames, abstraction thresholds, decision policies, knowledge sources and
the production-rule base, and offers ontology (frame) reasoning services such as
inheritance of slots, components, parameters, facts, faults and rules."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

from .rules import Rule, parse_rules

DEFAULT_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kb")


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class KnowledgeBase:
    def __init__(self, kb_dir: str = DEFAULT_KB_DIR):
        self.kb_dir = kb_dir
        onto = _load_json(os.path.join(kb_dir, "ontology.json"))
        self.classes: Dict[str, dict] = onto["classes"]
        self.components: Dict[str, str] = onto["components"]
        self.parameters: Dict[str, dict] = onto["parameters"]
        self.assets: Dict[str, dict] = onto["assets"]
        self.facts: Dict[str, dict] = _load_json(os.path.join(kb_dir, "facts.json"))["facts"]
        self.faults: Dict[str, dict] = _load_json(os.path.join(kb_dir, "faults.json"))["faults"]
        self.causes: Dict[str, dict] = _load_json(os.path.join(kb_dir, "causes.json"))["causes"]
        self.actions: Dict[str, dict] = _load_json(os.path.join(kb_dir, "actions.json"))["actions"]
        self.thresholds: dict = _load_json(os.path.join(kb_dir, "thresholds.json"))
        self.policies: dict = _load_json(os.path.join(kb_dir, "policies.json"))
        self.sources: Dict[str, dict] = _load_json(os.path.join(kb_dir, "sources.json"))["sources"]
        with open(os.path.join(kb_dir, "rules.kb"), encoding="utf-8") as fh:
            self.rules: List[Rule] = parse_rules(fh.read())
        self.rule_index: Dict[str, Rule] = {r.id: r for r in self.rules}

    # ------------------------------------------------------------------
    # Frame / ontology services
    # ------------------------------------------------------------------
    def ancestors(self, cls: str) -> List[str]:
        """Class followed by its ancestors up to the root (is-a chain)."""
        chain = []
        while cls:
            if cls not in self.classes:
                raise KeyError(f"unknown class '{cls}'")
            chain.append(cls)
            cls = self.classes[cls]["parent"]
        return chain

    def is_a(self, cls: str, ancestor: str) -> bool:
        return ancestor in self.ancestors(cls)

    def subclasses(self, cls: str) -> List[str]:
        return [c for c in self.classes if c != cls and cls in self.ancestors(c)]

    def applies(self, domain: List[str], cls: str) -> bool:
        anc = set(self.ancestors(cls))
        return any(d in anc for d in domain)

    def class_slots(self, cls: str) -> Dict[str, dict]:
        slots: Dict[str, dict] = {}
        for c in reversed(self.ancestors(cls)):
            for k, v in self.classes[c].get("slots", {}).items():
                slots[k] = dict(v, defined_in=c)
        return slots

    def class_components(self, cls: str) -> List[dict]:
        out = []
        for c in reversed(self.ancestors(cls)):
            for comp in self.classes[c].get("components", []):
                out.append({"id": comp, "name": self.components.get(comp, comp), "inherited_from": c})
        return out

    def class_parameters(self, cls: str) -> List[str]:
        out: List[str] = []
        for c in reversed(self.ancestors(cls)):
            for p in self.classes[c].get("parameters", []):
                if p not in out:
                    out.append(p)
        return out

    def parameter_options(self, param: str, cls: str) -> List[dict]:
        opts = self.parameters[param].get("options", [])
        return [o for o in opts if self.applies(o.get("domain", ["Equipment"]), cls)]

    def facts_for(self, cls: str) -> Dict[str, dict]:
        return {k: v for k, v in self.facts.items() if self.applies(v["domain"], cls)}

    def faults_for(self, cls: str) -> Dict[str, dict]:
        return {k: v for k, v in self.faults.items() if self.applies(v["domain"], cls)}

    def rules_for(self, cls: str) -> List[Rule]:
        return [r for r in self.rules if self.applies(r.domain, cls)]

    def asset_frame(self, asset_id: str) -> dict:
        """Instance frame with inherited default slot values filled in."""
        if asset_id not in self.assets:
            raise KeyError(f"unknown asset '{asset_id}'")
        raw = self.assets[asset_id]
        cls = raw["class"]
        frame = {"id": asset_id, "class": cls, "class_label": self.classes[cls]["label"],
                 "ancestors": self.ancestors(cls), "defaults_used": []}
        for slot, spec in self.class_slots(cls).items():
            if slot in raw:
                frame[slot] = raw[slot]
            elif "default" in spec:
                frame[slot] = spec["default"]
                frame["defaults_used"].append(slot)
        for k, v in raw.items():
            if k not in frame:
                frame[k] = v
        if frame.get("iso_group") == "auto":
            g1 = self.thresholds["iso10816_3"]["group1_min_kw"]
            frame["iso_group"] = 1 if float(frame.get("power_kw", 0)) > g1 else 2
        return frame

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------
    def label(self, fact: str) -> str:
        if fact.startswith("fault:"):
            f = self.faults.get(fact[6:])
            return f["name"] if f else fact
        if fact.startswith("cause:"):
            c = self.causes.get(fact[6:])
            return c["name"] if c else fact
        if fact.startswith("safety:"):
            s = self.policies["safety_constraints"].get(fact[7:])
            return s["label"] if s else fact
        f = self.facts.get(fact)
        return f["label"] if f else fact

    def is_conclusion(self, fact: str) -> bool:
        return fact.split(":", 1)[0] in ("fault", "cause", "safety") and ":" in fact

    @lru_cache(maxsize=None)
    def concluding_rules(self, fact: str) -> tuple:
        return tuple(r for r in self.rules if r.conclusion == fact)

    # ------------------------------------------------------------------
    # Knowledge-base verification (consistency & completeness checks)
    # ------------------------------------------------------------------
    def validate(self) -> dict:
        errors, warnings = [], []
        safety = self.policies["safety_constraints"]
        used_facts, concluded = set(), set()
        for r in self.rules:
            for d in r.domain:
                if d not in self.classes:
                    errors.append(f"{r.id}: unknown domain class '{d}'")
            for s in r.sources:
                if s not in self.sources:
                    errors.append(f"{r.id}: unknown source '{s}'")
            for f in r.premise_facts():
                used_facts.add(f)
                if not self._known_fact(f):
                    errors.append(f"{r.id}: premise refers to unknown fact '{f}'")
            if not self._known_fact(r.conclusion):
                errors.append(f"{r.id}: conclusion refers to unknown fact '{r.conclusion}'")
            concluded.add(r.conclusion)
            pre, _, name = r.conclusion.partition(":")
            if pre == "safety" and name not in safety:
                errors.append(f"{r.id}: unknown safety constraint '{name}'")
        for fid, f in self.faults.items():
            if f"fault:{fid}" not in concluded:
                errors.append(f"fault '{fid}' has no diagnostic rule")
            for a in f["actions"]:
                if a["id"] not in self.actions:
                    errors.append(f"fault '{fid}': unknown action '{a['id']}'")
            for s in f.get("sources", []):
                if s not in self.sources:
                    errors.append(f"fault '{fid}': unknown source '{s}'")
            for d in f["domain"]:
                if d not in self.classes:
                    errors.append(f"fault '{fid}': unknown domain '{d}'")
        for cid, c in self.causes.items():
            if f"cause:{cid}" not in concluded:
                warnings.append(f"cause '{cid}' is never concluded by a rule")
            if c.get("prevent") and c["prevent"] not in self.actions:
                errors.append(f"cause '{cid}': unknown preventive action '{c['prevent']}'")
        for fid, f in self.facts.items():
            if fid not in used_facts:
                warnings.append(f"fact '{fid}' is not used by any rule")
            for p in f.get("derived_from", []):
                if p not in self.parameters:
                    errors.append(f"fact '{fid}': unknown parameter '{p}'")
            for d in f["domain"]:
                if d not in self.classes:
                    errors.append(f"fact '{fid}': unknown domain '{d}'")
        for sc, spec in safety.items():
            for a in spec["mandatory_actions"]:
                if a not in self.actions:
                    errors.append(f"safety constraint '{sc}': unknown action '{a}'")
        cycle = self._find_cycle()
        if cycle:
            errors.append("rule dependency cycle: " + " -> ".join(cycle))
        for aid, a in self.assets.items():
            if a["class"] not in self.classes:
                errors.append(f"asset '{aid}': unknown class")
            elif self.classes[a["class"]].get("abstract"):
                errors.append(f"asset '{aid}': instance of abstract class")
        return {"errors": errors, "warnings": warnings,
                "stats": self.stats()}

    def _known_fact(self, f: str) -> bool:
        pre, sep, name = f.partition(":")
        if not sep:
            return f in self.facts
        if pre == "fault":
            return name in self.faults
        if pre == "cause":
            return name in self.causes
        if pre == "safety":
            return name in self.policies["safety_constraints"]
        return False

    def dependency_graph(self) -> Dict[str, set]:
        g: Dict[str, set] = {}
        for r in self.rules:
            for f in r.premise_facts():
                g.setdefault(f, set()).add(r.conclusion)
        return g

    def _find_cycle(self) -> Optional[List[str]]:
        g = self.dependency_graph()
        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {}
        stack: List[str] = []

        def visit(n: str) -> Optional[List[str]]:
            colour[n] = GREY
            stack.append(n)
            for m in g.get(n, ()):
                if colour.get(m, WHITE) == GREY:
                    return stack[stack.index(m):] + [m]
                if colour.get(m, WHITE) == WHITE:
                    res = visit(m)
                    if res:
                        return res
            stack.pop()
            colour[n] = BLACK
            return None

        for node in list(g):
            if colour.get(node, WHITE) == WHITE:
                res = visit(node)
                if res:
                    return res
        return None

    def stats(self) -> dict:
        types: Dict[str, int] = {}
        for r in self.rules:
            types[r.type] = types.get(r.type, 0) + 1
        concrete = [c for c, v in self.classes.items() if not v.get("abstract")]
        return {
            "classes": len(self.classes),
            "equipment_types": len(concrete),
            "components": len(self.components),
            "parameters": len(self.parameters),
            "evidence_facts": len(self.facts),
            "faults": len(self.faults),
            "causes": len(self.causes),
            "actions": len(self.actions),
            "rules": len(self.rules),
            "rules_by_type": types,
            "assets": len(self.assets),
            "sources": len(self.sources),
        }


_DEFAULT: Optional[KnowledgeBase] = None


def load_default_kb() -> KnowledgeBase:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KnowledgeBase()
    return _DEFAULT
