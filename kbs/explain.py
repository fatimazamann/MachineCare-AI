"""Explanation facility: HOW a conclusion was reached (proof tree with the
certainty-factor arithmetic), plus text renderings used by the CLI, the web UI
and the report."""

from __future__ import annotations

from typing import List, Optional

from . import certainty as cfm


def how(kb, wm, fact: str, depth: int = 0, max_depth: int = 4, _seen: Optional[set] = None) -> dict:
    seen = set() if _seen is None else _seen
    node = {"fact": fact, "label": kb.label(fact), "cf": round(wm.cf(fact), 4),
            "status": cfm.label(wm.cf(fact), kb.policies["status_thresholds"]), "support": []}
    if fact in seen or depth > max_depth:
        node["truncated"] = True
        return node
    seen = seen | {fact}
    for c in wm.contributions(fact):
        entry = {"source": c.source_id, "type": c.source_type, "cf": round(c.cf, 4), "detail": c.detail}
        if c.source_type == "rule" and c.source_id in kb.rule_index:
            rule = kb.rule_index[c.source_id]
            conds = []
            for cond in rule.conditions:
                children = []
                for f in cond.facts:
                    if wm.known(f):
                        children.append(how(kb, wm, f, depth + 1, max_depth, seen))
                conds.append({"text": cond.text(), "cf": round(cond.evaluate(wm.cf), 4), "facts": children})
            entry["rule"] = {"id": rule.id, "type": rule.type, "text": rule.text(), "note": rule.note,
                             "rule_cf": rule.cf, "premise_cf": c.premise_cf, "sources": rule.sources,
                             "conditions": conds}
            premise_list = ", ".join("{:+.2f}".format(x["cf"]) for x in conds)
            entry["arithmetic"] = "{:+.2f} × min({}) = {:+.3f}".format(rule.cf, premise_list, c.cf)
        node["support"].append(entry)
    node["combination"] = wm.combination_steps(fact)
    return node


def how_text(node: dict, indent: int = 0) -> List[str]:
    pad = "    " * indent
    lines = [f"{pad}{node['label']}  [CF {node['cf']:+.2f}, {node['status']}]"]
    for s in node["support"]:
        if "rule" in s:
            r = s["rule"]
            lines.append(f"{pad}  ← rule {r['id']} ({r['type']}): {r['text']}")
            if r.get("note"):
                lines.append(f"{pad}      “{r['note']}”")
            lines.append(f"{pad}      contribution = {s['arithmetic']}")
            for c in r["conditions"]:
                for child in c["facts"]:
                    lines.extend(how_text(child, indent + 2))
        else:
            lines.append(f"{pad}  ← {s['type']} ({s['source']}): CF {s['cf']:+.2f}  {s['detail']}")
    if len(node.get("combination", [])) > 1:
        steps = "; ".join(f"{st['before']:+.2f} ⊕ {st['cf']:+.2f} = {st['after']:+.2f}" for st in node["combination"])
        lines.append(f"{pad}  combined (MYCIN): {steps}")
    return lines


def narrative(result: dict) -> str:
    """Short natural-language summary of a diagnosis result."""
    asset = result["asset"]
    parts = []
    faults = [f for f in result["faults"] if f["cf"] >= 0.2]
    active = [s for s in result["safety"] if s["active"]]
    pending = [s for s in result["safety"] if not s["active"]]
    if active:
        parts.append("SAFETY: " + "; ".join(f"{s['label']} (CF {s['cf']:.2f})" for s in active) + ".")
    if pending:
        parts.append("Possible safety issue – verify now: " + "; ".join(
            f"{s['label']} (CF {s['cf']:.2f}, below the activation level)" for s in pending) + ".")
    if not faults:
        parts.append(f"No fault reached the 'possible' level for {asset['id']} with the evidence provided.")
    else:
        top = faults[0]
        parts.append(f"Most likely fault on {asset['id']} ({asset['class_label']}): {top['name']} — "
                     f"{top['status'].lower()} (CF {top['cf']:.2f}), urgency {top['urgency']}.")
        if top.get("causes"):
            c0 = top["causes"][0]
            word = "Probable" if c0["cf"] >= 0.5 else "Possible"
            parts.append(f"{word} root cause: {c0['name']} (CF {c0['cf']:.2f}).")
        if len(faults) > 1:
            parts.append("Other candidates: " + ", ".join(f"{f['name']} ({f['cf']:.2f})" for f in faults[1:4]) + ".")
    if result["conflicts"]:
        parts.append(f"{len(result['conflicts'])} conflicting evidence item(s) were resolved by certainty combination.")
    if result.get("rule_conflicts"):
        parts.append(f"{len(result['rule_conflicts'])} hypothesis(es) had supporting and opposing rules (e.g. explaining away).")
    if result["ambiguities"]:
        a = result["ambiguities"][0]
        parts.append(f"Ambiguous between {a['a_name']} and {a['b_name']}; answer the suggested questions to discriminate.")
    return " ".join(parts)
