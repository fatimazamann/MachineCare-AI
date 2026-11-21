"""Inference engine: forward chaining with an agenda and conflict resolution,
goal-driven backward chaining, and question generation."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from . import certainty as cfm
from .rules import Rule
from .working_memory import Contribution, WorkingMemory

AskFn = Callable[[str, List[Tuple[str, str]]], Optional[float]]


class InferenceEngine:
    def __init__(self, kb, wm: WorkingMemory, asset_class: str):
        self.kb = kb
        self.wm = wm
        self.cls = asset_class
        self.rules: List[Rule] = kb.rules_for(asset_class)   # rules inherited along the is-a chain
        self.order = {r.id: i for i, r in enumerate(self.rules)}
        self.threshold = kb.policies.get("fire_threshold", 0.2)
        self.trace: List[dict] = []
        self.cycles = 0
        self.asked: List[dict] = []

    # ================================================================
    #  FORWARD CHAINING  (recognise – resolve – act cycle)
    # ================================================================
    def _activation(self, rule: Rule) -> Optional[dict]:
        """Is this rule ready to fire (or to update / withdraw what it concluded earlier)?"""
        pcf = rule.premise_cf(self.wm.cf)
        current = self.wm.contribution(rule.conclusion, rule.id)
        if pcf > self.threshold:
            contrib = cfm.propagate(rule.cf, pcf)
            if current is None or abs(current.cf - contrib) > 1e-6:
                return {"rule": rule, "premise_cf": pcf, "contribution": contrib,
                        "op": "fire" if current is None else "update"}
        elif current is not None:
            # a premise no longer holds (e.g. a negative rule lowered a fault): withdraw the conclusion
            return {"rule": rule, "premise_cf": pcf, "contribution": 0.0, "op": "retract",
                    "reason": "premise no longer holds"}
        return None

    def agenda(self) -> List[dict]:
        """Conflict set, ordered by the conflict-resolution strategy:
        1. salience (safety 100 > diagnostic 50 > cause 30),
        2. specificity (more conditions first),
        3. order of the rule in the rule base."""
        acts = [a for a in (self._activation(r) for r in self.rules) if a]
        acts.sort(key=lambda a: (-a["rule"].salience, -a["rule"].specificity, self.order[a["rule"].id]))
        return acts

    @staticmethod
    def _why_chosen(top: dict, other: Optional[dict]) -> str:
        if other is None:
            return "only rule in the conflict set"
        r1, r2 = top["rule"], other["rule"]
        if r1.salience != r2.salience:
            return f"higher priority ({r1.type} {r1.salience} > {r2.type} {r2.salience})"
        if r1.specificity != r2.specificity:
            return f"more specific ({r1.specificity} vs {r2.specificity} conditions)"
        return "earlier in the rule base"

    def forward_chain(self, max_cycles: int = 2000) -> int:
        fired = 0
        while self.cycles < max_cycles:
            agenda = self.agenda()                      # 1. recognise
            if not agenda:
                break
            self.cycles += 1
            act = agenda[0]                             # 2. resolve (pick the best rule)
            rule = act["rule"]
            before = self.wm.cf(rule.conclusion)
            if act["op"] == "retract":                  # 3. act
                self.wm.retract(rule.conclusion, rule.id)
            else:
                self.wm.put(rule.conclusion, Contribution(
                    source_id=rule.id, source_type="rule", cf=act["contribution"],
                    detail=rule.note, rule_cf=rule.cf, premise_cf=round(act["premise_cf"], 4),
                    cycle=self.cycles))
            fired += 1
            self.trace.append({
                "cycle": self.cycles, "mode": "forward", "op": act["op"], "rule": rule.id, "type": rule.type,
                "salience": rule.salience, "premise_cf": round(act["premise_cf"], 4), "rule_cf": rule.cf,
                "contribution": round(act["contribution"], 4), "conclusion": rule.conclusion,
                "cf_before": round(before, 4), "cf_after": round(self.wm.cf(rule.conclusion), 4),
                "conflict_set": [a["rule"].id for a in agenda[:8]], "conflict_set_size": len(agenda),
                "chosen_because": self._why_chosen(act, agenda[1] if len(agenda) > 1 else None),
                "reason": act.get("reason", ""),
            })
        return fired

    # ================================================================
    #  BACKWARD CHAINING  (goal-driven, MYCIN style)
    # ================================================================
    def backward_chain(self, goal: str, ask: Optional[AskFn] = None, depth: int = 0,
                       why: Tuple[Tuple[str, str], ...] = (), max_depth: int = 6,
                       _visited: Optional[set] = None) -> float:
        """Try to establish `goal`. Unknown askable facts are obtained through
        `ask(fact, why_chain)`; without a callback they are left unknown."""
        visited = _visited if _visited is not None else set()
        if goal in visited or depth > max_depth:
            return self.wm.cf(goal)
        visited.add(goal)

        if not self.kb.is_conclusion(goal):            # a leaf fact: look it up or ask the user
            if self.wm.known(goal):
                return self.wm.cf(goal)
            info = self.kb.facts.get(goal, {})
            if ask and info.get("askable") and self.kb.applies(info.get("domain", []), self.cls):
                answer = ask(goal, list(why))
                self.asked.append({"fact": goal, "why": list(why), "answer": answer})
                if answer is not None:
                    self.wm.put(goal, Contribution("answer:" + goal, "operator", float(answer),
                                                   detail="answered during backward chaining"))
                    self.trace.append({"cycle": self.cycles, "mode": "backward", "op": "ask", "fact": goal,
                                       "answer_cf": answer, "why": [w[1] for w in why]})
            return self.wm.cf(goal)

        # a conclusion: try every rule that concludes it
        rules = [r for r in self.kb.concluding_rules(goal) if self.kb.applies(r.domain, self.cls)]
        rules.sort(key=lambda r: (-r.salience, -r.specificity, -abs(r.cf)))
        for rule in rules:
            chain = why + ((goal, rule.id),)
            satisfiable = True
            for cond in rule.conditions:
                for fact in cond.facts:
                    if self.kb.is_conclusion(fact) or not self.wm.known(fact):
                        self.backward_chain(fact, ask, depth + 1, chain, max_depth, visited)
                    if cond.kind == "any" and cond.evaluate(self.wm.cf) > self.threshold:
                        break
                if cond.evaluate(self.wm.cf) <= self.threshold:
                    satisfiable = False
                    break  # this premise failed: no need to ask about the rest of this rule
            pcf = rule.premise_cf(self.wm.cf)
            before = self.wm.cf(goal)
            if satisfiable and pcf > self.threshold:
                self.wm.put(goal, Contribution(rule.id, "rule", cfm.propagate(rule.cf, pcf), rule.note,
                                               rule_cf=rule.cf, premise_cf=round(pcf, 4), cycle=self.cycles))
                self.trace.append({"cycle": self.cycles, "mode": "backward", "op": "fire", "rule": rule.id,
                                   "type": rule.type, "premise_cf": round(pcf, 4), "rule_cf": rule.cf,
                                   "contribution": round(cfm.propagate(rule.cf, pcf), 4), "conclusion": goal,
                                   "cf_before": round(before, 4), "cf_after": round(self.wm.cf(goal), 4)})
        return self.wm.cf(goal)

    # ================================================================
    #  QUESTION GENERATION  (which unknown fact is most worth asking?)
    # ================================================================
    def _unknown(self, fact: str) -> bool:
        if self.kb.is_conclusion(fact):
            return self.wm.cf(fact) == 0
        return not self.wm.known(fact)

    def _condition_status(self, cond) -> str:
        """satisfied | unknown (could still become true) | failed."""
        v = cond.evaluate(self.wm.cf)
        if v > self.threshold:
            return "satisfied"
        if any(self._unknown(f) for f in cond.facts):
            return "unknown"  # for ANY(...) one unknown item is enough to keep it open
        return "failed"

    def suggest_questions(self, max_questions: int = 6) -> List[dict]:
        """Score every unknown fact by how much it could change the diagnosis.

        score(fact) = Σ over rules that still could fire and need the fact:
                      |rule CF| × hypothesis weight × readiness × share / (number of unknown conditions)
          hypothesis weight = (0.35 + 0.65 × CF of the fault) × (0.6 + 0.08 × severity)
          readiness         = 0.6^(unknown conditions − 1), × 0.7 for a condition that needs a measurement
          share             = 0.8 for an item inside ANY(...), else 1.0
        ×1.5 when the fact concerns only one of two nearly tied faults (it discriminates between them).
        """
        t = self.kb.policies["status_thresholds"]
        faults = {f[6:]: cf for f, cf in self.wm.facts("fault:").items()}
        applicable = self.kb.faults_for(self.cls)
        hyps: Dict[str, float] = {}
        live = {k: v for k, v in faults.items() if v > 0.05}
        if live:
            for fid, cf in live.items():
                if cf < t["confirmed"]:
                    hyps[fid] = cf
        else:
            hyps = {fid: 0.0 for fid in applicable}  # screening mode: nothing suspected yet

        ranked = sorted(hyps.items(), key=lambda kv: -kv[1])
        ambiguous = set()
        if len(ranked) >= 2 and ranked[0][1] > 0 and ranked[0][1] - ranked[1][1] < self.kb.policies["ambiguity_margin"]:
            ambiguous = {ranked[0][0], ranked[1][0]}

        scores: Dict[str, float] = {}
        reasons: Dict[str, List[dict]] = {}
        support: Dict[str, set] = {}
        for fid, hcf in hyps.items():
            sev = applicable.get(fid, {}).get("severity", 3)
            w = (0.35 + 0.65 * max(hcf, 0.0)) * (0.6 + 0.08 * sev)
            for rule in self.kb.concluding_rules(f"fault:{fid}"):
                if not self.kb.applies(rule.domain, self.cls):
                    continue
                statuses = [self._condition_status(c) for c in rule.conditions]
                if "failed" in statuses:
                    continue
                unknown_conds = [c for c, st in zip(rule.conditions, statuses) if st == "unknown"]
                if not unknown_conds:
                    continue
                readiness = 0.6 ** (len(unknown_conds) - 1)
                for c in unknown_conds:
                    if not any(self.kb.facts.get(f, {}).get("askable") for f in c.facts):
                        readiness *= 0.7
                for c in unknown_conds:
                    share = 0.8 if c.kind == "any" else 1.0
                    for f in c.facts:
                        if self.kb.is_conclusion(f) or self.wm.known(f):
                            continue
                        info = self.kb.facts.get(f, {})
                        if not (info.get("askable") or info.get("derived_from")):
                            continue
                        s = abs(rule.cf) * w * readiness * share / len(unknown_conds)
                        scores[f] = scores.get(f, 0.0) + s
                        support.setdefault(f, set()).add(fid)
                        reasons.setdefault(f, []).append({"rule": rule.id, "fault": fid, "rule_cf": rule.cf,
                                                          "hyp_cf": round(hcf, 3)})
        for f, fs in support.items():
            if ambiguous and len(fs & ambiguous) == 1:
                scores[f] *= 1.5

        # root-cause questions for a fault that is already probable (they select the preventive action)
        for fid, fcf in faults.items():
            if fcf < t["probable"]:
                continue
            for rule in self.rules:
                if rule.type != "cause" or f"fault:{fid}" not in rule.premise_facts():
                    continue
                if self.wm.cf(rule.conclusion) >= t["probable"]:
                    continue
                statuses = [self._condition_status(c) for c in rule.conditions]
                if "failed" in statuses:
                    continue
                for c, st in zip(rule.conditions, statuses):
                    if st != "unknown":
                        continue
                    for f in c.facts:
                        info = self.kb.facts.get(f, {})
                        if self.kb.is_conclusion(f) or not info.get("askable"):
                            continue
                        s = 0.45 * abs(rule.cf) * fcf
                        if s > scores.get(f, 0):
                            scores[f] = s
                            support.setdefault(f, set()).add(fid)
                            reasons.setdefault(f, []).insert(0, {"rule": rule.id, "fault": fid, "rule_cf": rule.cf,
                                                                 "hyp_cf": round(fcf, 3), "cause": rule.conclusion[6:]})

        out = []
        for f, s in sorted(scores.items(), key=lambda kv: -kv[1])[:max_questions]:
            info = self.kb.facts[f]
            rs = sorted(reasons[f], key=lambda r: -abs(r["rule_cf"]))
            top = rs[0]
            fname = self.kb.faults[top["fault"]]["name"]
            if top.get("cause"):
                why = (f"Root cause: rule {top['rule']} would establish '{self.kb.causes[top['cause']]['name']}' "
                       f"for '{fname}' (CF {top['hyp_cf']:+.2f}) – this selects the preventive action.")
            elif top["hyp_cf"] > 0:
                why = (f"Rule {top['rule']} needs this to {'support' if top['rule_cf'] > 0 else 'weaken'} "
                       f"'{fname}' (currently CF {top['hyp_cf']:+.2f}; rule CF {top['rule_cf']:+.2f}).")
            else:
                why = f"Screening: rule {top['rule']} uses this to test '{fname}' (rule CF {top['rule_cf']:+.2f})."
            if len(support[f]) > 1:
                why += " Also relevant to: " + ", ".join(self.kb.faults[x]["name"] for x in sorted(support[f]) if x != top["fault"]) + "."
            if ambiguous and len(support[f] & ambiguous) == 1:
                why += " It discriminates between the two leading hypotheses."
            q_type = "ask" if info.get("askable") else "measure"
            out.append({
                "fact": f, "type": q_type, "question": info["question"], "label": info["label"],
                "kind": info["kind"], "group": info.get("group"),
                "measure": [self.kb.parameters[p]["label"] for p in info.get("derived_from", [])] if q_type == "measure" else [],
                "params": info.get("derived_from", []),
                "score": round(s, 3), "why": why, "hypotheses": sorted(support[f]), "rules": [r["rule"] for r in rs],
            })
        return out
