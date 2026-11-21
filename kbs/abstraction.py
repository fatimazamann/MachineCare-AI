"""Data abstraction: converts raw numeric measurements into qualitative facts
with certainty factors (the 'feature-abstraction' step of heuristic
classification).  All limits come from data/kb/thresholds.json so that the
knowledge can be edited without touching code."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Abstraction:
    fact: str
    band_cf: float           # CF of the qualitative band itself
    detail: str              # human-readable derivation
    params: Dict[str, float] = field(default_factory=dict)


def _num(m: dict, key: str) -> Optional[float]:
    v = m.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def abstract_measurements(kb, frame: dict, measurements: dict) -> List[Abstraction]:
    th = kb.thresholds
    cls = frame["class"]
    out: List[Abstraction] = []
    m = measurements or {}

    def add(fact, cf, detail, **params):
        if kb.applies(kb.facts[fact]["domain"], cls):
            out.append(Abstraction(fact, cf, detail, params))

    # ---------------- rotating machinery ----------------
    if kb.is_a(cls, "RotatingMachine"):
        v = _num(m, "vibration_velocity")
        if v is not None:
            iso = th["iso10816_3"]
            group = int(frame.get("iso_group", 2))
            foundation = frame.get("foundation", "rigid")
            ab, bc, cd = iso[f"group{group}"][foundation]
            zone = "A" if v < ab else "B" if v < bc else "C" if v < cd else "D"
            zcf = iso["zone_cf"][zone]
            limits = f"A/B {ab}, B/C {bc}, C/D {cd} mm/s (ISO 10816-3 group {group}, {foundation})"
            add("high_vibration", zcf, f"{_fmt(v)} mm/s RMS → Zone {zone}; {limits}", vibration_velocity=v)
            sev = 0.95 if zone == "D" else (-0.5 if zone == "C" else -0.9)
            add("vibration_severe", sev, f"{_fmt(v)} mm/s RMS → Zone {zone} (Zone D starts at {cd} mm/s)", vibration_velocity=v)
        spec = m.get("vib_spectrum")
        if spec and spec in kb.facts:
            label = next((o["label"] for o in kb.parameters["vib_spectrum"]["options"] if o["value"] == spec), spec)
            add(spec, 0.9, f"Vibration analyst reports: {label}")
        bt = _num(m, "bearing_temp")
        if bt is not None:
            b = th["bearing_temp"]
            cf = 0.85 if bt >= b["high_c"] else (0.4 if bt >= b["watch_c"] else -0.7)
            add("bearing_overheating", cf, f"Bearing {_fmt(bt)} °C (watch {b['watch_c']}, alarm {b['high_c']} °C)", bearing_temp=bt)
            ccf = 0.95 if bt >= b["critical_c"] else (-0.3 if bt >= b["high_c"] else -0.8)
            add("bearing_temp_critical", ccf, f"Bearing {_fmt(bt)} °C vs. trip {b['critical_c']} °C", bearing_temp=bt)

    # ---------------- centrifugal pump ----------------
    if kb.is_a(cls, "CentrifugalPump"):
        p = th["pump"]
        na, nr = _num(m, "npsh_available"), _num(m, "npsh_required")
        if na is not None and nr:
            ratio = na / nr
            cf = 0.95 if ratio < p["npsh_ratio_critical"] else (0.6 if ratio < p["npsh_ratio_low"] else -0.7)
            add("npsh_margin_low", cf,
                f"NPSHa/NPSH3 = {_fmt(na)}/{_fmt(nr)} = {ratio:.2f} (critical < {p['npsh_ratio_critical']}, low < {p['npsh_ratio_low']})",
                npsh_available=na, npsh_required=nr)
        h = _num(m, "head_pct_rated")
        if h is not None:
            cf = 0.85 if h < p["head_low"] else (0.4 if h < p["head_slightly_low"] else -0.7)
            add("low_head", cf, f"Head {_fmt(h)} % of curve value (low < {p['head_low']} %)", head_pct_rated=h)
        c = _num(m, "current_pct_normal")
        if c is not None:
            lo = 0.85 if c < p["current_low"] else (0.4 if c < 80 else -0.6)
            add("low_power_draw", lo, f"Current {_fmt(c)} % of normal (low < {p['current_low']} %)", current_pct_normal=c)
        s = _num(m, "seal_leak_rate")
        if s is not None:
            cf = 0.9 if s > p["seal_leak_excessive"] else (0.5 if s > p["seal_leak_elevated"] else -0.8)
            add("seal_leakage", cf, f"Seal leakage {_fmt(s)} drops/min (elevated > {p['seal_leak_elevated']}, excessive > {p['seal_leak_excessive']})", seal_leak_rate=s)

    # ---------------- induction motor ----------------
    if kb.is_a(cls, "InductionMotor"):
        mt = th["motor"]
        vs = [_num(m, k) for k in ("v_ab", "v_bc", "v_ca")]
        if all(x is not None for x in vs):
            avg = sum(vs) / 3
            if avg > 0:
                vu = max(abs(x - avg) for x in vs) / avg * 100
                desc = f"V = {_fmt(vs[0],0)}/{_fmt(vs[1],0)}/{_fmt(vs[2],0)} V, avg {_fmt(avg,1)} V → unbalance {vu:.2f} % (NEMA MG 1: max deviation / average)"
                if vu >= mt["vu_max_pct"]:
                    cf, sev = 0.95, 0.95
                elif vu >= mt["vu_high_pct"]:
                    cf, sev = 0.85, -0.8
                elif vu >= mt["vu_derate_pct"]:
                    cf, sev = 0.6, -0.9
                else:
                    cf, sev = -0.8, -0.95
                add("voltage_unbalance", cf, desc + f"; derate above {mt['vu_derate_pct']} %", unbalance_pct=round(vu, 2))
                add("voltage_unbalance_severe", sev, desc + f"; limit {mt['vu_max_pct']} %", unbalance_pct=round(vu, 2))
                rated = float(frame.get("rated_voltage") or 400)
                pct = avg / rated * 100
                lcf = 0.85 if pct < mt["supply_low_pct"] else (0.3 if pct < 95 else -0.7)
                add("supply_voltage_low", lcf, f"Average {_fmt(avg,1)} V = {pct:.1f} % of rated {rated:.0f} V (low < {mt['supply_low_pct']} %)", supply_pct=round(pct, 1))
                if min(vs) < mt["phase_loss_voltage_ratio"] * max(vs):
                    add("phase_loss", 0.9, f"Lowest line voltage {_fmt(min(vs),0)} V < {mt['phase_loss_voltage_ratio']:.0%} of highest {_fmt(max(vs),0)} V")
        cs = [_num(m, k) for k in ("i_a", "i_b", "i_c")]
        if all(x is not None for x in cs) and max(cs) > 0:
            avg = sum(cs) / 3
            cu = max(abs(x - avg) for x in cs) / avg * 100 if avg else 0
            desc = f"I = {_fmt(cs[0],1)}/{_fmt(cs[1],1)}/{_fmt(cs[2],1)} A, avg {_fmt(avg,1)} A → current unbalance {cu:.1f} %"
            ccf = 0.85 if cu >= mt["cu_high_pct"] else (0.5 if cu >= mt["cu_moderate_pct"] else -0.7)
            add("current_unbalance", ccf, desc + f" (high ≥ {mt['cu_high_pct']} %)", current_unbalance_pct=round(cu, 1))
            if min(cs) < mt["phase_loss_current_ratio"] * max(cs):
                add("phase_loss", 0.95, f"{desc}; one phase carries < {mt['phase_loss_current_ratio']:.0%} of the highest phase current")
            else:
                add("phase_loss", -0.9, f"{desc}; all three phases carry current")
        lf = _num(m, "load_pct_fla")
        if lf is not None:
            cf = 0.95 if lf >= mt["load_severe_pct"] else 0.8 if lf >= mt["load_over_pct"] else 0.3 if lf >= mt["load_normal_max_pct"] else -0.8
            add("motor_overloaded", cf, f"Current {_fmt(lf)} % of FLA (overload ≥ {mt['load_over_pct']} %)", load_pct_fla=lf)
        wt = _num(m, "winding_temp")
        if wt is not None:
            cls_ins = str(frame.get("insulation_class", "F"))
            lim = mt["class_limit_c"].get(cls_ins, 155)
            marg = mt["winding_margin_k"]
            if wt >= lim:
                oh, cr = 0.95, 0.95
            elif wt >= lim - marg:
                oh, cr = 0.8, -0.5
            elif wt >= lim - 2 * marg:
                oh, cr = 0.3, -0.9
            else:
                oh, cr = -0.8, -0.95
            d = f"Winding {_fmt(wt)} °C vs. class {cls_ins} limit {lim} °C (alarm from {lim - marg} °C)"
            add("motor_overheating", oh, d, winding_temp=wt)
            add("winding_temp_critical", cr, d, winding_temp=wt)
        sb = _num(m, "sideband_db")
        if sb is not None:
            cf = 0.95 if sb >= mt["sideband_multiple_db"] else 0.8 if sb >= mt["sideband_broken_db"] else 0.4 if sb >= mt["sideband_early_db"] else -0.8
            add("rotor_sidebands", cf, f"Sidebands at {_fmt(sb)} dB (healthy < {mt['sideband_early_db']} dB, broken bars ≥ {mt['sideband_broken_db']} dB)", sideband_db=sb)

    # ---------------- CNC milling (AI4I 2020 definitions) ----------------
    if kb.is_a(cls, "CNCMillingMachine"):
        a = th["ai4i"]
        air, proc = _num(m, "air_temp_k"), _num(m, "process_temp_k")
        rpm, tq, wear = _num(m, "rot_speed_rpm"), _num(m, "torque_nm"), _num(m, "tool_wear_min")
        variant = str(m.get("product_variant") or frame.get("product_variant") or "M").upper()
        if air is not None and proc is not None:
            d = proc - air
            add("low_temp_difference", 0.95 if d < a["hdf_dt_k"] else -0.95,
                f"ΔT = {_fmt(proc,1)} − {_fmt(air,1)} = {d:.2f} K (HDF if < {a['hdf_dt_k']} K)", delta_t=round(d, 2))
        if rpm is not None:
            add("low_spindle_speed", 0.95 if rpm < a["hdf_rpm"] else -0.95,
                f"Speed {_fmt(rpm,0)} rpm (HDF if < {a['hdf_rpm']} rpm)", rot_speed_rpm=rpm)
        if rpm is not None and tq is not None:
            power = tq * rpm * 2 * math.pi / 60
            out_rng = power < a["pwf_min_w"] or power > a["pwf_max_w"]
            add("power_out_of_range", 0.95 if out_rng else -0.95,
                f"P = {_fmt(tq,1)} Nm × {_fmt(rpm,0)} rpm × 2π/60 = {power:.0f} W (valid {a['pwf_min_w']}–{a['pwf_max_w']} W)", power_w=round(power))
            if not out_rng:
                near = (power < a["pwf_min_w"] * (1 + a["power_near_frac"])) or (power > a["pwf_max_w"] * (1 - a["power_near_frac"]))
                add("power_near_limit", 0.8 if near else -0.6, f"P = {power:.0f} W; within {a['power_near_frac']:.0%} of a limit: {'yes' if near else 'no'}", power_w=round(power))
        if wear is not None and tq is not None:
            limit = a["osf_limit"].get(variant, 12000)
            strain = wear * tq
            add("overstrain", 0.95 if strain > limit else -0.95,
                f"Wear × torque = {_fmt(wear,0)} min × {_fmt(tq,1)} Nm = {strain:.0f} min·Nm (variant {variant} limit {limit})", strain=round(strain))
            if strain <= limit:
                near = strain >= a["osf_near_frac"] * limit
                add("overstrain_approaching", 0.8 if near else -0.6, f"{strain:.0f} min·Nm = {strain / limit:.0%} of limit {limit}", strain=round(strain))
        if wear is not None:
            add("tool_wear_critical", 0.95 if wear >= a["twf_start_min"] else -0.9,
                f"Tool wear {_fmt(wear,0)} min (TWF window {a['twf_start_min']}–{a['twf_end_min']} min)", tool_wear_min=wear)

    # ---------------- train air compressor / APU (MetroPT-3) ----------------
    if kb.is_a(cls, "AirCompressor"):
        p = th["apu"]
        duty = _num(m, "duty_pct_30")
        if duty is not None:
            cf = 0.95 if duty >= p["duty_strong_pct"] else 0.5 if duty >= p["duty_moderate_pct"] else 0.0 if duty >= 30 else -0.8
            add("continuous_loading", cf, f"Loaded {_fmt(duty,0)} % of the last 30 min (normal ≈ 15 %; leak ≥ {p['duty_strong_pct']} %)", duty_pct_30=duty)
        cur = _num(m, "current_mean_a_30")
        if cur is not None:
            add("high_motor_load", 0.85 if cur >= p["current_high_a"] else -0.6,
                f"Average current {_fmt(cur,1)} A (high ≥ {p['current_high_a']} A; normal ≈ 2 A)", current_mean_a_30=cur)
        oil = _num(m, "oil_temp_c_max_30")
        if oil is not None:
            cf = 0.9 if oil >= p["oil_very_high_c"] else 0.6 if oil >= p["oil_high_c"] else -0.7
            add("oil_temp_high", cf, f"Oil up to {_fmt(oil,1)} °C (high ≥ {p['oil_high_c']} °C, very high ≥ {p['oil_very_high_c']} °C)", oil_temp_c_max_30=oil)
        tp3 = _num(m, "panel_pressure_min_bar_30")
        if tp3 is not None:
            cf = 0.95 if tp3 < p["low_pressure_bar"] else 0.6 if tp3 < p["cut_in_bar"] else -0.5
            add("panel_pressure_low", cf, f"Lowest panel pressure {_fmt(tp3,2)} bar (cut-in {p['cut_in_bar']} bar, alarm {p['low_pressure_bar']} bar)", panel_pressure_min_bar_30=tp3)
        dv = _num(m, "dryer_dv_pressure_bar_30")
        if dv is not None and duty is not None and duty >= p["duty_moderate_pct"]:
            add("dryer_purge_high", 0.8 if dv >= p["dryer_dv_high_bar"] else -0.8,
                f"Dryer discharge pressure {_fmt(dv,2)} bar while loaded (high ≥ {p['dryer_dv_high_bar']} bar)", dryer_dv_pressure_bar_30=dv)
        lps = m.get("lps_active")
        if lps not in (None, ""):
            on = str(lps).lower() in ("yes", "1", "true", "1.0")
            add("low_pressure_alarm", 0.95 if on else -0.9, "LPS alarm " + ("active: pressure below 7 bar" if on else "not active"))
    return out
