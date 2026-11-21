"""Facade that wires the knowledge base, data abstraction, inference engine,
explanation facility and maintenance recommender together."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from . import certainty as cfm
from .abstraction import abstract_measurements
from .engine import InferenceEngine
from .explain import how, how_text, narrative
from .kb import KnowledgeBase, load_default_kb
from .recommender import Recommender
from .working_memory import Contribution, WorkingMemory

REPORTERS = ("operator", "technician", "sensor")


class DiagnosisSystem:
    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or load_default_kb()
        self.recommender = Recommender(self.kb)

    # ------------------------------------------------------------------
    def _observation_cf(self, value, reporter: str) -> Optional[float]:
        """yes/no answer -> CF scaled by the reliability of whoever reported it."""
        rel = self.kb.policies["source_reliability"]
        r = rel.get(reporter, rel["operator"])
        if value is None or value == "" or value == "unknown":
            return None
        if isinstance(value, bool):
            return r if value else -r
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("yes", "true", "y", "1"):
                return r
            if v in ("no", "false", "n", "0"):
                return -r
            return None
        return cfm.clamp(float(value))  # explicit certainty in [-1, 1]

    def build_memory(self, asset_id: str, observations: Optional[dict] = None, measurements: Optional[dict] = None,
                     reporter: str = "operator"):
        kb = self.kb
        frame = kb.asset_frame(asset_id)
        cls = frame["class"]
        wm = WorkingMemory()
        rel = kb.policies["source_reliability"]
        warnings, inputs = [], []

        # (1) asset register (instance frame slots).  Values stored for the asset are certain;
        #     values inherited as class DEFAULTS are assumptions with a low CF that evidence can override.
        default_cf = kb.policies.get("default_slot_cf", 0.3)
        for slot in ("hazardous_fluid",):
            if slot in kb.facts and kb.applies(kb.facts[slot]["domain"], cls) and slot in frame:
                if slot in frame.get("defaults_used", []):
                    mag, src, how_ = default_cf, "default", f"Default inherited from class frame: {slot} = {frame[slot]}"
                else:
                    mag, src, how_ = rel["asset_register"], "asset", f"Asset register: {slot} = {frame[slot]}"
                cf = mag if frame[slot] else -mag
                wm.put(slot, Contribution(f"{src}:{slot}", src, cf, how_))
                inputs.append({"fact": slot, "source": src, "cf": cf})

        # (2) sensor / test measurements -> qualitative facts (data abstraction)
        abstractions = []
        for i, ab in enumerate(abstract_measurements(kb, frame, measurements or {})):
            cf = cfm.clamp(ab.band_cf * rel["sensor"])
            wm.put(ab.fact, Contribution(f"sensor:{ab.fact}:{i}", "sensor", cf, ab.detail))
            abstractions.append({"fact": ab.fact, "label": kb.label(ab.fact), "band_cf": ab.band_cf, "cf": cf,
                                 "detail": ab.detail})

        # (3) operator / technician observations
        for fact, value in (observations or {}).items():
            rep = reporter
            if isinstance(value, dict):
                rep = value.get("reporter", reporter)
                value = value.get("cf", value.get("value"))
            if fact not in kb.facts:
                warnings.append(f"Unknown observation '{fact}' ignored")
                continue
            if not kb.applies(kb.facts[fact]["domain"], cls):
                warnings.append(f"'{fact}' does not apply to {cls}; ignored")
                continue
            cf = self._observation_cf(value, rep)
            if cf is None:
                continue
            wm.put(fact, Contribution(f"{rep}:{fact}", rep if rep in REPORTERS else "operator", cf,
                                      f"Reported by {rep}: {'yes' if cf > 0 else 'no'}"))
            inputs.append({"fact": fact, "source": rep, "cf": cf})
        return frame, wm, abstractions, inputs, warnings

    def candidate_faults(self, wm: WorkingMemory, cls: str, top: int = 3) -> List[str]:
        """Faults with a rule that uses some evidence already believed true (for backward chaining)."""
        positive = {f for f, cf in wm.evidence_facts().items() if cf > 0.2}
        scored = []
        for fid, f in self.kb.faults_for(cls).items():
            hits = sum(1 for r in self.kb.concluding_rules(f"fault:{fid}") if r.cf > 0 and positive & set(r.premise_facts()))
            if hits:
                scored.append((hits, f["severity"], fid))
        return [fid for _, _, fid in sorted(scored, reverse=True)[:top]]

    # ------------------------------------------------------------------
    def diagnose(self, asset_id: str, observations: Optional[dict] = None, measurements: Optional[dict] = None,
                 reporter: str = "operator", policy: str = "balanced", ask: Optional[Callable] = None,
                 verify: Optional[list] = None) -> dict:
        t0 = time.perf_counter()
        frame, wm, abstractions, inputs, warnings = self.build_memory(asset_id, observations, measurements, reporter)
        engine = InferenceEngine(self.kb, wm, frame["class"])
        engine.forward_chain()                                   # data-driven step
        if ask is not None:                                      # goal-driven step: ask about uncertain faults
            goals = list(verify or [])
            if not goals:
                live = sorted(((f[6:], cf) for f, cf in wm.facts("fault:").items()
                               if 0.05 < cf < self.kb.policies["status_thresholds"]["confirmed"]), key=lambda x: -x[1])
                goals = [fid for fid, _ in live[:3]] or self.candidate_faults(wm, frame["class"])
            for g in goals:
                engine.backward_chain(f"fault:{g}", ask=ask)
            engine.forward_chain()
        elif verify:
            for g in verify:
                engine.backward_chain(f"fault:{g}")
            engine.forward_chain()
        return self._result(frame, wm, engine, abstractions, inputs, warnings, policy, t0)

    # ------------------------------------------------------------------
    def _result(self, frame, wm, engine, abstractions, inputs, warnings, policy, t0) -> dict:
        kb = self.kb
        th = kb.policies["status_thresholds"]
        faults = {f[6:]: cf for f, cf in wm.facts("fault:").items()}
        causes = {c[6:]: cf for c, cf in wm.facts("cause:").items()}
        safety = {}
        links: Dict[str, set] = {}
        for s, cf in wm.facts("safety:").items():
            sid = s[7:]
            rules = [c.source_id for c in wm.contributions(s)]
            safety[sid] = {"cf": cf, "rules": rules}
            for rid in rules:
                r = kb.rule_index.get(rid)
                if r:
                    for f in r.premise_facts():
                        if f.startswith("fault:"):
                            links.setdefault(sid, set()).add(f[6:])
        plan = self.recommender.recommend(frame, {k: v for k, v in faults.items() if v > 0}, causes, safety, policy, links)
        plan_by_fault = {p["fault"]: p for p in plan["fault_plans"]}

        evidence = wm.evidence_facts()
        fault_list = []
        for fid, cf in sorted(faults.items(), key=lambda kv: -kv[1]):
            f = kb.faults[fid]
            linked_causes = []
            for cid, ccf in causes.items():
                for c in wm.contributions(f"cause:{cid}"):
                    r = kb.rule_index.get(c.source_id)
                    if r and f"fault:{fid}" in r.premise_facts() and ccf > 0:
                        linked_causes.append({"id": cid, "name": kb.causes[cid]["name"], "cf": round(ccf, 4)})
                        break
            linked_causes.sort(key=lambda x: -x["cf"])
            fp = plan_by_fault.get(fid, {})
            entry = {
                "id": fid, "name": f["name"], "cf": round(cf, 4), "status": cfm.label(cf, th),
                "severity": f["severity"], "safety_critical": f["safety_critical"],
                "component": kb.components.get(f["component"], f["component"]),
                "description": f["description"],
                "risk": fp.get("risk"), "urgency": fp.get("urgency", "—"), "urgency_label": fp.get("urgency_label", ""),
                "deferred": fp.get("deferred", False),
                "rules": [{"id": c.source_id, "cf": round(c.cf, 4), "type": c.source_type} for c in wm.contributions(f"fault:{fid}")],
                "causes": linked_causes,
                "actions": fp.get("actions", []),
            }
            if cf >= th["possible"]:
                tree = how(kb, wm, f"fault:{fid}")
                entry["how"] = tree
                entry["how_text"] = how_text(tree)
            fault_list.append(entry)

        reported = [f for f in fault_list if f["cf"] >= th["possible"]]
        ambiguities = []
        if len(reported) >= 2 and reported[0]["cf"] - reported[1]["cf"] < kb.policies["ambiguity_margin"]:
            a, b = reported[0], reported[1]
            ambiguities.append({"a": a["id"], "a_name": a["name"], "b": b["id"], "b_name": b["name"],
                                "difference": round(a["cf"] - b["cf"], 4)})
        conflicts, rule_conflicts = [], []
        for c in wm.conflicts(kb.policies["conflict_threshold"]):
            c["label"] = kb.label(c["fact"])
            (rule_conflicts if kb.is_conclusion(c["fact"]) else conflicts).append(c)
        result = {
            "asset": frame,
            "policy": policy,
            "inputs": inputs,
            "abstractions": abstractions,
            "evidence": [{"fact": f, "label": kb.label(f), "cf": round(cf, 4), "kind": kb.facts.get(f, {}).get("kind"),
                          "sources": [c.to_dict() for c in wm.contributions(f)]}
                         for f, cf in sorted(evidence.items(), key=lambda kv: -abs(kv[1]))],
            "faults": fault_list,
            "causes": [{"id": c, "name": kb.causes[c]["name"], "cf": round(cf, 4)} for c, cf in
                       sorted(causes.items(), key=lambda kv: -kv[1]) if cf > 0],
            "safety": [{"id": k, "label": kb.policies["safety_constraints"][k]["label"], "cf": round(v["cf"], 4),
                        "rules": v["rules"], "active": v["cf"] >= kb.policies["safety_activation_cf"]}
                       for k, v in safety.items() if v["cf"] > 0],
            "conflicts": conflicts,
            "rule_conflicts": rule_conflicts,
            "ambiguities": ambiguities,
            "questions": engine.suggest_questions(),
            "plan": plan,
            "trace": engine.trace,
            "asked": engine.asked,
            "warnings": warnings,
            "stats": {"cycles": engine.cycles, "rule_firings": len([t for t in engine.trace if t.get("op") in ("fire", "update")]),
                      "rules_considered": len(engine.rules), "facts_in_memory": len(wm.facts()),
                      "time_ms": round((time.perf_counter() - t0) * 1000, 2)},
        }
        result["summary"] = narrative(result)
        return result
