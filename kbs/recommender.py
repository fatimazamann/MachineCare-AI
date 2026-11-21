"""Maintenance recommender: turns diagnosed faults, root causes and safety
constraints into a prioritised, explained maintenance plan.

Stakeholder conflicts are resolved in three layers:
  1. SAFETY FIRST – an active safety constraint is a hard rule: its mandatory
     actions come first and no stakeholder profile can postpone them.
  2. URGENCY – risk = severity (1–5) × CF of the fault decides how soon to act.
  3. TRADE-OFFS – among the possible actions for a fault, the one with the best
     weighted score is chosen:
         score = Σ weight(profile, criterion) × value(action, criterion)
     criteria: effectiveness, safety gain, cost, downtime, production continuity;
     the weights come from the selected stakeholder profile.
"""

from __future__ import annotations

from typing import Dict, List

STAGE_ORDER = {"containment": 0, "verify": 1, "corrective": 2, "preventive": 3, "monitor": 4}
URGENCY_ORDER = {"IMMEDIATE": 0, "URGENT": 1, "SCHEDULED": 2, "MONITOR": 3}
WHEN = {"IMMEDIATE": "Immediately (after isolation)", "URGENT": "Within 24 h",
        "SCHEDULED": "Next planned stop (≤ 7 days)", "MONITOR": "When convenient"}


class Recommender:
    def __init__(self, kb):
        self.kb = kb
        self.p = kb.policies

    # ------------------------------------------------------------------ the weighted score
    def production_score(self, action: dict, frame: dict) -> float:
        """1 = no production loss … 0 = a critical machine stops with no backup."""
        if not action.get("requires_shutdown"):
            return 1.0
        if frame.get("has_standby"):
            return 0.8
        return {"A": 0.0, "B": 0.3, "C": 0.6}.get(frame.get("criticality", "B"), 0.3)

    def score(self, action_id: str, effectiveness: float, frame: dict, weights: Dict[str, float]) -> dict:
        a = self.kb.actions[action_id]
        values = {
            "effectiveness": effectiveness,
            "safety": a.get("safety_gain", 0.0),
            "cost": 1 - a.get("cost", 3) / self.p["cost_norm"],
            "downtime": 1 - min(a.get("downtime_h", 0), self.p["downtime_norm_h"]) / self.p["downtime_norm_h"],
            "production": self.production_score(a, frame),
        }
        total = sum(weights[k] * values[k] for k in weights)
        return {"score": round(total, 4), "criteria": {k: round(v, 3) for k, v in values.items()},
                "weighted": {k: round(weights[k] * values[k], 4) for k in weights}}

    def feasible(self, action_id: str, frame: dict) -> bool:
        return not (self.kb.actions[action_id].get("requires_standby") and not frame.get("has_standby"))

    # ------------------------------------------------------------------ urgency
    def urgency(self, fault_id: str, cf: float, safety_faults: set) -> tuple:
        f = self.kb.faults[fault_id]
        risk = round(f["severity"] * cf, 3)
        u = self.p["urgency"]
        th = self.p["status_thresholds"]
        if fault_id in safety_faults or (f.get("safety_critical") and cf >= th["probable"]):
            level = "IMMEDIATE"
        elif f.get("safety_critical") and cf >= th["possible"]:
            level = "URGENT"  # precaution: a possible safety-critical fault is checked within 24 h
        elif risk >= u["urgent_risk"]:
            level = "URGENT"
        elif risk >= u["scheduled_risk"]:
            level = "SCHEDULED"
        else:
            level = "MONITOR"
        return risk, level

    # ------------------------------------------------------------------ the plan
    def recommend(self, frame: dict, faults: Dict[str, float], causes: Dict[str, float],
                  safety: Dict[str, dict], profile_id: str = "balanced", fault_safety_links: Dict[str, set] = None) -> dict:
        profiles = self.p["profiles"]
        profile = profiles.get(profile_id, profiles["balanced"])
        weights = profile["weights"]
        th = self.p["status_thresholds"]
        tradeoffs: List[str] = []
        steps: Dict[str, dict] = {}

        def add_step(aid, stage, when, reason, score=None, urgency="MONITOR"):
            if not self.feasible(aid, frame):
                return
            a = self.kb.actions[aid]
            s = steps.get(aid)
            if s is None:
                steps[aid] = {"action": aid, "name": a["name"], "stage": stage, "when": when, "urgency": urgency,
                              "reasons": [reason], "score": score, "requires_shutdown": a.get("requires_shutdown", False),
                              "downtime_h": a.get("downtime_h", 0), "cost": a.get("cost", 0), "steps": a.get("steps", [])}
            else:  # same action needed for two reasons: keep the earliest stage and the highest urgency
                if reason not in s["reasons"]:
                    s["reasons"].append(reason)
                if STAGE_ORDER[stage] < STAGE_ORDER[s["stage"]]:
                    s["stage"], s["when"] = stage, when
                if URGENCY_ORDER[urgency] < URGENCY_ORDER[s["urgency"]]:
                    s["urgency"], s["when"] = urgency, when
                if score is not None and (s["score"] is None or score > s["score"]):
                    s["score"] = score

        # 1) safety constraints (hard) --------------------------------------
        active = {k: v for k, v in safety.items() if v["cf"] >= self.p["safety_activation_cf"]}
        safety_faults = set()
        for sc_id in active:
            spec = self.p["safety_constraints"][sc_id]
            for aid in spec["mandatory_actions"]:
                add_step(aid, "containment", "NOW – before any other work", f"Safety constraint: {spec['label']}",
                         urgency="IMMEDIATE")
            safety_faults |= (fault_safety_links or {}).get(sc_id, set())
        if active:
            tradeoffs.append("Safety constraint active → production and cost preferences are overridden "
                             "(safety is a hard rule, not a weighted criterion).")

        # 2) one plan per fault ---------------------------------------------
        fault_plans = []
        for fid, cf in sorted(faults.items(), key=lambda kv: -kv[1]):
            if cf < th["possible"]:
                continue
            f = self.kb.faults[fid]
            risk, level = self.urgency(fid, cf, safety_faults)
            stopped_note = None
            if active and cf >= th["probable"] and URGENCY_ORDER[level] > URGENCY_ORDER["URGENT"]:
                level, stopped_note = "URGENT", "brought forward: the machine is already stopped by a safety rule"
            ranked = []
            for a in f["actions"]:
                if not self.feasible(a["id"], frame):
                    continue
                sc = self.score(a["id"], a["effectiveness"], frame, weights)
                ranked.append({"action": a["id"], "name": self.kb.actions[a["id"]]["name"], "stage": a["stage"],
                               "effectiveness": a["effectiveness"], **sc})
            ranked.sort(key=lambda r: (STAGE_ORDER[r["stage"]], -r["score"]))
            plan = {"fault": fid, "name": f["name"], "cf": round(cf, 4), "severity": f["severity"], "risk": risk,
                    "urgency": level, "urgency_label": self.p["urgency"]["labels"][level], "note": stopped_note,
                    "deferred": False, "actions": ranked}
            when = "During the current safety stop" if active and level in ("IMMEDIATE", "URGENT") else WHEN[level]
            by_stage = {st: [r for r in ranked if r["stage"] == st] for st in STAGE_ORDER}

            if cf >= th["probable"]:
                # the main repair must actually fix the fault (effectiveness ≥ 0.7), then the best score wins
                corrective = [r for r in by_stage["corrective"] if r["effectiveness"] >= 0.7] or by_stage["corrective"]
                best = max(corrective, key=lambda r: r["score"]) if corrective else None
                needs_stop = bool(best and self.kb.actions[best["action"]].get("requires_shutdown"))
                # production profile may postpone a non-safety repair on a machine without a standby
                defer = (profile.get("defer_corrective") and level == "URGENT" and not active and needs_stop
                         and risk < self.p["urgency"]["non_deferrable_risk"] and not frame.get("has_standby"))

                if level in ("IMMEDIATE", "URGENT"):
                    for r in by_stage["containment"]:
                        add_step(r["action"], "containment", "Now", f"{f['name']} ({level})", r["score"], level)
                    if frame.get("has_standby") and needs_stop:
                        add_step("switch_to_standby", "containment", "Before the repair",
                                 f"{f['name']}: keep production running on the standby unit", urgency=level)
                        tradeoffs.append(f"{f['name']}: standby unit available → duty transferred so the repair "
                                         f"can be done {when.lower()} without production loss.")
                if cf < th["confirmed"]:
                    for r in by_stage["verify"][:1]:
                        add_step(r["action"], "verify", "Before corrective work", f"Confirm {f['name']} (CF {cf:.2f})", r["score"], level)
                if defer:
                    plan["deferred"] = True
                    plan["urgency_label"] = "Deferred to next planned stop with mitigation"
                    for aid in ("reduce_load", "increase_monitoring"):
                        add_step(aid, "containment" if aid == "reduce_load" else "monitor", "Until the planned stop",
                                 f"Interim mitigation for deferred {f['name']}", urgency="URGENT")
                    add_step(best["action"], "corrective", "Next planned stop (deferred)", f"{f['name']} (deferred)", best["score"], "SCHEDULED")
                    tradeoffs.append(f"{f['name']}: production-continuity profile → repair deferred to the next planned stop "
                                     f"(risk {risk:.2f} < {self.p['urgency']['non_deferrable_risk']}, no safety constraint, "
                                     f"no standby); load reduction and closer monitoring until then.")
                elif best:
                    add_step(best["action"], "corrective", when, f"{f['name']} (CF {cf:.2f})", best["score"], level)
                for r in by_stage["preventive"][:1]:
                    add_step(r["action"], "preventive", "After repair / next planned stop", f"Prevent recurrence of {f['name']}", r["score"])
                for r in by_stage["monitor"][:1]:
                    add_step(r["action"], "monitor", "Ongoing", f"Track {f['name']}", r["score"])
            else:  # only possible: check it at the next inspection
                check = by_stage["verify"] or by_stage["monitor"] or sorted(ranked, key=lambda r: -r["score"])
                for r in check[:1]:
                    add_step(r["action"], r["stage"], "At next inspection", f"Check possible {f['name']} (CF {cf:.2f})", r["score"], "MONITOR")
            fault_plans.append(plan)

        # 3) remove the root cause --------------------------------------------
        for cid, ccf in sorted(causes.items(), key=lambda kv: -kv[1]):
            if ccf < th["probable"]:
                continue
            c = self.kb.causes[cid]
            if c.get("prevent") and c["prevent"] in self.kb.actions:
                add_step(c["prevent"], "preventive", "After repair / next planned stop",
                         f"Remove root cause: {c['name']} (CF {ccf:.2f})")

        ordered = sorted(steps.values(), key=lambda s: (STAGE_ORDER[s["stage"]], URGENCY_ORDER[s["urgency"]], -(s["score"] or 0)))
        for i, s in enumerate(ordered, 1):
            s["step"] = i

        overall = "IMMEDIATE" if active else "MONITOR"
        for fp in fault_plans:
            if fp["cf"] >= th["probable"]:
                effective = "SCHEDULED" if fp["deferred"] else fp["urgency"]
                if URGENCY_ORDER[effective] < URGENCY_ORDER[overall]:
                    overall = effective
        shutdown_steps = [s for s in ordered if s["requires_shutdown"] and s["stage"] in ("containment", "corrective")
                          and "deferred" not in s["when"].lower()]
        downtime = round(sum(s["downtime_h"] for s in shutdown_steps), 1)
        cost_total = sum(s["cost"] for s in ordered if s["stage"] in ("containment", "verify", "corrective"))
        if frame.get("has_standby"):
            prod = "No production loss expected – duty can run on the standby unit."
        elif downtime == 0:
            prod = "No shutdown required for the immediate steps."
        else:
            prod = f"About {downtime} h of equipment downtime on a criticality-{frame.get('criticality', 'B')} asset without standby."
        return {
            "profile": {"id": profile_id, "label": profile["label"], "weights": weights},
            "overall_urgency": overall,
            "overall_label": self.p["urgency"]["labels"][overall],
            "safety_constraints": [{"id": k, "label": self.p["safety_constraints"][k]["label"], "cf": round(v["cf"], 4),
                                    "rules": v["rules"], "mandatory_actions": self.p["safety_constraints"][k]["mandatory_actions"]}
                                   for k, v in active.items()],
            "fault_plans": fault_plans,
            "steps": ordered,
            "estimated_downtime_h": downtime,
            "cost_level": "Low" if cost_total <= 3 else "Medium" if cost_total <= 7 else "High",
            "production_impact": prod,
            "tradeoffs": tradeoffs,
        }
