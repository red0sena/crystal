"""Programmatic skill data for two public-suite skills our models score ~0 on.

home-policy: a household device state + a spoken request -> how a policy-following assistant handles it.
    Independent generator (own rooms, device kinds, names, policies, request wording and question wording);
    it does not import or copy the evaluation kit's generator, households or templates.
symbolic-chord: score notes around a target beat -> pitch-class set of the sounding chord.
    Synthetic progressions with voicings, inversions, omitted fifths and non-chord tones; no POP909 data.

Labels are exact by construction (label_origin "programmatic"). Rows follow the v3 text-row schema.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path("runs/champion-v1/skill-synth-v1")
SEED = 20261107


def digest(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def row(source, family, uid, state, instructions, options, answer, item, ordered=False, extra=None):
    assert answer in options and len(options) == len(set(options)) >= 2
    criteria = {o: None for o in options}
    out = {"id": f"{source}:{uid}", "original_id": f"{source}:{item}", "source": source, "family": family,
           "modality": "text", "state": state, "question": {"type": "choice", "instructions": instructions,
           "criteria": criteria}, "target": [float(o == answer) for o in options], "kind": "choice",
           "ordered": ordered, "item": f"{source}:{item}", "documents": [], "split": "train",
           "label_origin": "programmatic", "target_kind": "single",
           "annotation_basis": "exact label computed by a deterministic generator", "training_component": "skill-synth-v1"}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- home policy

ROOMS = ["kitchen", "living room", "dining room", "hallway", "bathroom", "master bedroom", "kids room", "guest bedroom",
         "office", "laundry", "garage", "basement", "attic", "porch", "balcony", "garden", "playroom", "library",
         "pantry", "sunroom", "studio", "gym", "cellar", "foyer", "loft", "terrace", "workshop", "den", "nursery", "patio"]
NAMES = {
    "light": ["Ceiling Light", "Floor Lamp", "Desk Lamp", "Strip Lights", "Wall Sconce", "Chandelier", "Night Light",
              "Spotlights", "Bedside Lamp", "Accent Light"],
    "appliance": ["Coffee Maker", "Space Heater", "Dehumidifier", "Fan Heater", "Toaster", "Rice Cooker", "Electric Blanket",
                  "Aquarium Pump", "Printer", "Monitor", "Kettle", "Humidifier", "Air Fryer", "Iron", "Mini Fridge"],
    "thermostat": ["Thermostat", "Climate Panel", "Heat Pump"],
    "speaker": ["Smart Speaker", "Soundbar", "Radio", "Stereo"],
    "lock": ["Front Door Lock", "Back Door Lock", "Gate Lock", "Garage Door Lock", "Shed Lock"],
    "blinds": ["Blinds", "Curtains", "Shutters", "Shades"],
    "fan": ["Ceiling Fan", "Tower Fan", "Desk Fan", "Exhaust Fan"],
}
WATTS = {"Coffee Maker": 900, "Space Heater": 1500, "Dehumidifier": 300, "Fan Heater": 2000, "Toaster": 850,
         "Rice Cooker": 600, "Electric Blanket": 90, "Aquarium Pump": 15, "Printer": 45, "Monitor": 30, "Kettle": 2200,
         "Humidifier": 35, "Air Fryer": 1400, "Iron": 1200, "Mini Fridge": 70}
LEVEL = {"light": ("brightness", 0, 100, "%"), "speaker": ("volume", 0, 100, ""), "blinds": ("position", 0, 100, "%"),
         "fan": ("speed", 1, 5, ""), "thermostat": ("temperature", None, None, "°C")}
STATUS = {"execute": "carry it out now", "execute_partial": "carry out only the available parts now",
          "schedule": "schedule it for later", "rule": "save it as an automation rule",
          "clarify": "ask the user a clarifying question", "no_change": "change nothing because it is already so",
          "unsupported": "refuse because the setting is not supported", "blocked": "refuse because household policy blocks it",
          "query": "just report the current state"}


def make_home(rng):
    rooms = rng.sample(ROOMS, rng.randint(4, 7))
    devices, used = [], set()
    def add(kind, room, name=None):
        name = name or rng.choice(NAMES[kind])
        did = f"{room.replace(' ', '_')}_{name.lower().replace(' ', '_')}"
        if did in used:
            return None
        used.add(did)
        d = {"id": did, "name": name, "room": room, "kind": kind, "available": rng.random() > .08}
        if kind == "lock":
            d["locked"] = rng.random() < .5
        elif kind == "blinds":
            d["position"] = rng.choice(range(0, 101, 10))
        else:
            d["power"] = rng.random() < .5
        if kind == "light":
            d["brightness"] = rng.choice(range(10, 101, 5))
            d["watts"] = rng.choice([5, 8, 10, 12, 15, 40, 60])
        elif kind == "appliance":
            d["watts"] = WATTS[name]
        elif kind == "speaker":
            d["volume"] = rng.choice(range(0, 101, 5))
            d["watts"] = 20
        elif kind == "fan":
            d["speed"] = rng.randint(1, 5)
            d["watts"] = 45
        elif kind == "thermostat":
            lo = rng.choice([10, 14, 16, 18])
            d.update(temperature=rng.randint(lo, lo + 8), min_temperature=lo, max_temperature=lo + rng.choice([8, 10, 12]))
        devices.append(d)
        return d
    for room in rooms:
        for kind in rng.sample(["light", "light", "appliance", "appliance", "speaker", "blinds", "fan"], rng.randint(2, 4)):
            add(kind, room)
    add("thermostat", rng.choice(rooms))
    for _ in range(rng.randint(1, 2)):
        add("lock", rng.choice(rooms))
    if rng.random() < .6:  # same display name in two rooms
        name = rng.choice(NAMES["light"])
        a, b = rng.sample(rooms, 2)
        add("light", a, name)
        add("light", b, name)
    ids = [d["id"] for d in devices]
    groups = {}
    for gname in rng.sample(["movie night", "morning", "bedtime", "away", "dinner", "reading", "party"], rng.randint(1, 3)):
        groups[gname] = rng.sample(ids, rng.randint(2, min(5, len(ids))))
    aliases = {}
    alias = rng.choice(["upstairs", "back of the house", "front rooms", "kids area", "the annex"])
    aliases[alias] = rng.sample(rooms, rng.choice([1, 2, 2]))
    hour, minute = rng.randint(6, 22), rng.choice([0, 10, 15, 30, 45])
    home = {"local_time": f"{hour:02d}:{minute:02d}", "outside_temperature": rng.randint(-5, 32), "rooms": rooms,
            "room_aliases": aliases, "devices": devices, "groups": groups,
            "last_mentioned_device": rng.choice(ids),
            "policy": {"multiple_devices": rng.choice(["all_or_nothing", "best_effort"]),
                       "security_mode": rng.choice(["home", "home", "away"]),
                       "unlocking_when_away": "not allowed", "unavailable_devices": "cannot be operated"},
            "schedules": [], "rules": []}
    return home


def dev(home, did):
    return next(d for d in home["devices"] if d["id"] == did)


def say(rng, *variants):
    return rng.choice(variants)


def onoff(v):
    return "on" if v else "off"


def intent(home, rng):
    """Returns (spec, request). spec kinds mirror STATUS; ops are [device, property, value]."""
    ds = home["devices"]
    powered = [d for d in ds if "power" in d and d["kind"] != "thermostat"]
    unique = [d for d in powered if sum(x["name"] == d["name"] for x in ds) == 1]
    dup_names = sorted({d["name"] for d in ds if sum(x["name"] == d["name"] for x in ds) > 1})
    choice = rng.choice(["toggle", "toggle", "room_kind", "group", "duplicate", "pronoun", "missing", "level", "level_range",
                         "multi", "sequence", "schedule", "rule", "lock", "query", "alias", "watts", "group_except"])
    if choice == "toggle" and unique:
        d = rng.choice(unique)
        if rng.random() < .2:
            d["available"] = False
        v = rng.random() < .5
        return {"kind": "execute", "ops": [[d["id"], "power", v]]}, say(
            rng, f"Turn {onoff(v)} the {d['name'].lower()}.", f"Can you switch the {d['name'].lower()} {onoff(v)}?",
            f"Please {'start' if v else 'stop'} the {d['name'].lower()}.", f"{d['name']} {onoff(v)}, please.")
    if choice == "room_kind":
        room = rng.choice(home["rooms"])
        kind = rng.choice(["light", "appliance", "fan"])
        targets = [d for d in powered if d["room"] == room and d["kind"] == kind]
        if targets:
            v = rng.random() < .5
            keep = rng.choice(targets) if len(targets) > 1 and rng.random() < .5 else None
            ops = [[d["id"], "power", v] for d in targets if d is not keep]
            plural = {"light": "lights", "appliance": "appliances", "fan": "fans"}[kind]
            req = f"Turn {onoff(v)} all the {plural} in the {room}" + (f" except the {keep['name'].lower()}." if keep else ".")
            return {"kind": "execute", "ops": ops}, req
    if choice == "group" and home["groups"]:
        g = rng.choice(sorted(home["groups"]))
        v = rng.random() < .5
        members = [x for x in home["groups"][g] if "power" in dev(home, x)]
        if members:
            return {"kind": "execute", "ops": [[x, "power", v] for x in members]}, say(
                rng, f"Switch {onoff(v)} the {g} group.", f"{'Activate' if v else 'Shut down'} everything in '{g}'.")
    if choice == "group_except" and home["groups"]:
        g = rng.choice(sorted(home["groups"]))
        members = [x for x in home["groups"][g] if "power" in dev(home, x)]
        if len(members) >= 2:
            keep = rng.choice(members)
            return {"kind": "execute", "ops": [[x, "power", False] for x in members if x != keep]}, \
                f"Turn off the {g} group but leave the {dev(home, keep)['name'].lower()} in the {dev(home, keep)['room']} alone."
    if choice == "duplicate" and dup_names:
        name = rng.choice(dup_names)
        return {"kind": "clarify", "candidates": [d["id"] for d in ds if d["name"] == name]}, f"Turn on the {name.lower()}."
    if choice == "pronoun":
        d = dev(home, home["last_mentioned_device"])
        if "power" in d:
            v = not d["power"] if rng.random() < .8 else d["power"]
            return {"kind": "execute", "ops": [[d["id"], "power", v]]}, say(rng, f"Turn it {onoff(v)}.", f"And switch that one {onoff(v)} as well.")
    if choice == "missing":
        name = rng.choice([n for n in ["Dishwasher", "Robot Vacuum", "Sauna", "Hot Tub", "Projector", "Wine Cooler"]])
        return {"kind": "clarify", "candidates": []}, f"Start the {name.lower()} in the {rng.choice(home['rooms'])}."
    if choice in ("level", "level_range"):
        cands = [d for d in ds if d["kind"] in LEVEL and sum(x["name"] == d["name"] for x in ds) == 1]
        if cands:
            d = rng.choice(cands)
            prop, lo, hi, unit = LEVEL[d["kind"]]
            if d["kind"] == "thermostat":
                lo, hi = d["min_temperature"], d["max_temperature"]
            if choice == "level_range":
                value = hi + rng.randint(1, 6) if rng.random() < .5 else lo - rng.randint(1, 6)
                return {"kind": "unsupported", "ops": [[d["id"], prop, value]]}, f"Set the {d['name'].lower()} {prop} to {value}{unit}."
            if rng.random() < .5:
                value = rng.randint(lo, hi)
                return {"kind": "execute", "ops": [[d["id"], prop, value]]}, f"Set the {d['name'].lower()} to {value}{unit}."
            step = rng.choice([1, 2]) if d["kind"] in ("fan", "thermostat") else rng.choice([10, 20, 30])
            up = rng.random() < .5
            value = d[prop] + (step if up else -step)
            if value > hi or value < lo:
                return {"kind": "unsupported", "ops": [[d["id"], prop, value]]}, \
                    f"Make the {d['name'].lower()} {step} {'higher' if up else 'lower'}."
            return {"kind": "execute", "ops": [[d["id"], prop, value]]}, f"Make the {d['name'].lower()} {step} {'higher' if up else 'lower'}."
    if choice == "multi" and len(unique) >= 2:
        a, b = rng.sample(unique, 2)
        a["available"], b["available"] = True, rng.random() > .5
        v = rng.random() < .5
        return {"kind": "execute", "ops": [[a["id"], "power", v], [b["id"], "power", v]]}, \
            f"Turn {onoff(v)} the {a['name'].lower()} and the {b['name'].lower()}."
    if choice == "sequence" and unique:
        d = rng.choice(unique)
        first = rng.random() < .5
        return {"kind": "execute", "ops": [[d["id"], "power", first], [d["id"], "power", not first]]}, \
            f"Switch the {d['name'].lower()} {onoff(first)} and then {onoff(not first)} again."
    if choice == "schedule" and unique:
        d = rng.choice(unique)
        hour = rng.randint(0, 23)
        v = rng.random() < .5
        return {"kind": "schedule", "at": f"{hour:02d}:30", "ops": [[d["id"], "power", v]]}, \
            f"At {hour:02d}:30, turn {onoff(v)} the {d['name'].lower()}."
    if choice == "rule":
        therms = [d for d in ds if d["kind"] == "thermostat"]
        t = therms[0]
        temp = rng.randint(t["min_temperature"], t["max_temperature"])
        below = rng.randint(0, 15)
        return {"kind": "rule", "condition": ["outside_temperature", "below", below],
                "ops": [[t["id"], "temperature", temp]]}, f"Whenever it drops below {below}°C outside, set the {t['name'].lower()} to {temp}°C."
    if choice == "lock":
        locks = [d for d in ds if d["kind"] == "lock"]
        d = rng.choice(locks)
        unlock = rng.random() < .6
        req = f"{'Unlock' if unlock else 'Lock'} the {d['name'].lower()}."
        if unlock and home["policy"]["security_mode"] == "away":
            return {"kind": "blocked", "ops": [[d["id"], "locked", False]], "reason": "away mode"}, req
        return {"kind": "execute", "ops": [[d["id"], "locked", not unlock]]}, req
    if choice == "watts":
        room = rng.choice(home["rooms"])
        limit = rng.choice([50, 100, 500, 1000])
        targets = [d for d in powered if d["room"] == room and d["kind"] == "appliance" and d["watts"] > limit]
        if targets:
            return {"kind": "execute", "ops": [[d["id"], "power", False] for d in targets]}, \
                f"In the {room}, switch off any appliance drawing more than {limit} W."
    if choice == "alias":
        alias, rooms = next(iter(home["room_aliases"].items()))
        lights = [d for d in powered if d["room"] in rooms and d["kind"] == "light"]
        if lights:
            if len(rooms) > 1:
                return {"kind": "clarify", "candidates": [d["id"] for d in lights]}, f"Turn off the light in the {alias}."
            if len(lights) == 1:
                return {"kind": "execute", "ops": [[lights[0]["id"], "power", False]]}, f"Turn off the light in the {alias}."
    d = rng.choice([x for x in ds if x["kind"] in ("thermostat", "lock", "speaker", "blinds")])
    prop = {"thermostat": "temperature", "lock": "locked", "speaker": "volume", "blinds": "position"}[d["kind"]]
    return {"kind": "query", "target": d["id"], "property": prop}, f"What is the {d['name'].lower()} {prop} right now?"


def resolve(home, spec):
    """Policy semantics:
    - an operation on an unavailable device cannot run. A single-device request is then blocked; a multi-device
      request is blocked under all_or_nothing and runs only the available devices under best_effort.
    - operations run in the stated order, so a later operation on the same property wins.
    - a request that changes nothing is no_change. Schedules/rules change nothing now.
    referenced = devices the correctly interpreted request points at (none when it must be clarified)."""
    final = copy.deepcopy(home)
    status, changed = spec["kind"], []
    referenced = sorted({o[0] for o in spec.get("ops", [])} | ({spec["target"]} if "target" in spec else set()))
    if status == "execute":
        devices = {o[0] for o in spec["ops"]}
        down = {d for d in devices if not dev(home, d)["available"]}
        if down and (len(devices) == 1 or home["policy"]["multiple_devices"] == "all_or_nothing"):
            return {"status": "blocked", "referenced": referenced, "changed": [], "final": final}
        if down:
            status = "execute_partial"
        for did, prop, value in spec["ops"]:
            if did not in down:
                dev(final, did)[prop] = value
        changed = sorted([did, prop, dev(final, did)[prop]] for did, prop in {(o[0], o[1]) for o in spec["ops"]}
                         if dev(final, did)[prop] != dev(home, did)[prop])
        if not changed:
            status = "no_change"
    elif status == "schedule":
        final["schedules"].append({"at": spec["at"], "ops": spec["ops"]})
    elif status == "rule":
        final["rules"].append({"condition": spec["condition"], "ops": spec["ops"]})
    elif status == "clarify":
        referenced = []
    return {"status": status, "referenced": referenced, "changed": changed, "final": final}


def outcome_text(result):
    if result["status"] == "schedule":
        return "nothing changes now; a scheduled action is stored"
    if result["status"] == "rule":
        return "nothing changes now; an automation rule is stored"
    if not result["changed"]:
        return "no device changes"
    return "; ".join(f"{d} {p} -> {json.dumps(v)}" for d, p, v in result["changed"])


def home_rows(n_scenarios, seed=SEED):
    source, rows = "home-policy-synth", []
    for i in range(n_scenarios):
        rng = random.Random(digest(seed, "home", i))
        home = make_home(rng)
        spec, request = intent(home, rng)
        for _ in range(4):  # the status-query fallback otherwise dominates
            if spec["kind"] != "query" or rng.random() < .3:
                break
            spec, request = intent(home, rng)
        result = resolve(home, spec)
        state = {**home, "request": request}
        item = digest(seed, "home", i)[:16]
        acting = set(result["referenced"])
        qs = []
        qs.append(("handling", say(rng, "What should the assistant do with this request?", "How should the request be handled?",
                                   "Pick the correct way to respond to the request."),
                   list(STATUS.values()), STATUS[result["status"]]))
        ids = [d["id"] for d in home["devices"]]
        picks = sorted(acting) + rng.sample([x for x in ids if x not in acting], min(len(ids) - len(acting), max(2, len(acting))))
        for did in picks:
            qs.append((f"target:{did}", say(rng, f"Does the request, correctly interpreted, refer to {did}?",
                                            f"Is {did} one of the devices this request resolves to?"),
                       ["yes", "no"], "yes" if did in acting else "no"))
        out = outcome_text(result)
        distract = {"no device changes", "nothing changes now; a scheduled action is stored",
                    "nothing changes now; an automation rule is stored"}
        for o in spec.get("ops", [])[:2]:
            for v in ([True, False] if isinstance(o[2], bool) else [o[2], dev(home, o[0]).get(o[1])]):
                distract.add(f"{o[0]} {o[1]} -> {json.dumps(v)}")
        opts = [out] + sorted(x for x in distract if x != out)[:5]
        qs.append(("outcome", "What changes immediately after the correct handling?", opts, out))
        probe = rng.choice(home["devices"])
        prop = next(p for p in ("power", "locked", "temperature", "position") if p in probe)
        val = probe[prop]
        wrong = (not val) if isinstance(val, bool) else val + rng.choice([-2, -1, 1, 2])
        qs.append(("state", f"Before anything is done, what is the {prop} of {probe['id']}?",
                   [json.dumps(val), json.dumps(wrong), "not listed"], json.dumps(val)))
        for k, (did, prop, value) in enumerate(spec.get("ops", [])[:2]):
            after = dev(result["final"], did).get(prop)
            opts = list(dict.fromkeys([json.dumps(after), json.dumps(value), json.dumps(dev(home, did).get(prop)), "not applicable"]))
            if len(opts) >= 2:
                qs.append((f"after:{k}", f"Right after correct handling, what is {prop} of {did}?", opts, json.dumps(after)))
        for role, instr, opts, ans in qs:
            rows.append(row(source, "home-policy", f"{item}:{role}", state, instr, opts, ans, item,
                            extra={"scenario": spec["kind"], "role": role.split(":")[0]}))
    return rows


# ---------------------------------------------------------------- symbolic chords

QUALITIES = {"": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
             "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
             "mMaj7": (0, 3, 7, 11), "aug7": (0, 4, 8, 10)}
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OTHER = {"add9": (0, 2, 4, 7), "6sus4": (0, 5, 7, 9), "7sus4": (0, 5, 7, 10), "maj9no5": (0, 2, 4, 11)}


def vocabulary():
    names = {}
    for root in range(12):
        for q, ivs in QUALITIES.items():
            pcs = tuple(sorted((root + i) % 12 for i in ivs))
            names.setdefault(pcs, []).append(NOTE[root] + q)
    return names


VOCAB = vocabulary()
CHORD_SETS = sorted(VOCAB)


def option_text(pcs):
    return f"{' / '.join(VOCAB[pcs])} = pitch classes {list(pcs)}"


SCALE = (0, 2, 4, 5, 7, 9, 11)
DIATONIC = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m"), (11, "dim"), (7, "7"), (0, "maj7"), (2, "m7"),
            (5, "maj7"), (9, "m7"), (11, "m7b5"), (7, "sus4"), (2, "sus2"), (8, "aug"), (11, "dim7"), (0, "mMaj7"), (4, "aug7")]


def chord_events(rng, key, start, end):
    """A progression covering [start, end) with chord changes on 2- or 4-beat boundaries."""
    events, t = [], start
    while t < end:
        dur = rng.choice([2, 4, 4])
        if rng.random() < .08:
            pcs = tuple(sorted((key + rng.randrange(12) + i) % 12 for i in rng.choice(list(OTHER.values()))))
            label = pcs if pcs in VOCAB else "Other"
        elif rng.random() < .05:
            pcs, label = (), "NoChord"
        else:
            deg, q = rng.choice(DIATONIC) if rng.random() < .85 else (rng.randrange(12), rng.choice(list(QUALITIES)))
            root = (key + deg) % 12
            pcs = tuple(sorted((root + i) % 12 for i in QUALITIES[q]))
            label = pcs
        events.append((t, t + dur, pcs, label))
        t += dur
    return events


def notes_for(rng, key, events):
    notes = []
    for s, e, pcs, label in events:
        if not pcs:
            continue
        root_pc = pcs[0]
        tones = list(pcs)
        if len(tones) >= 4 and rng.random() < .3:  # omit the fifth of a seventh chord sometimes
            fifth = (tones[0] + 7) % 12
            if fifth in tones and len(tones) > 3:
                tones.remove(fifth)
        bass = rng.choice(tones) if rng.random() < .25 else root_pc
        notes.append((36 + bass + 12 * rng.choice([0, 1]), s, e - rng.choice([0, .25, .5])))
        style = rng.choice(["block", "arp", "broken"])
        t = s
        while t < e:
            if style == "block":
                for pc in tones:
                    notes.append((48 + pc + 12 * rng.choice([0, 1]), t, min(e, t + rng.choice([1, 2]))))
                t += rng.choice([1, 2])
            else:
                pc = rng.choice(tones)
                notes.append((48 + pc + 12 * rng.choice([0, 1]), t, t + .5))
                t += .5
        t = s
        while t < e:  # melody: mostly chord tones, some passing tones
            pc = rng.choice(tones) if rng.random() < .7 else (key + rng.choice(SCALE)) % 12
            dur = rng.choice([.5, 1, 1, 2])
            notes.append((72 + pc - 12 * rng.choice([0, 0, 1]), t, min(e, t + dur)))
            t += dur
    out = []
    for pitch, s, e in notes:
        if e > s:
            out.append({"pitch": pitch, "pitch_class": pitch % 12, "start_beat": round(s, 4), "end_beat": round(e, 4)})
    return sorted(out, key=lambda n: (n["start_beat"], n["pitch"]))


def chord_rows(n_items, seed=SEED):
    source, rows = "symbolic-chord-synth", []
    for i in range(n_items):
        rng = random.Random(digest(seed, "chord", i))
        key = rng.randrange(12)
        events = chord_events(rng, key, 0, 32)
        target = rng.choice(range(8, 24))
        sounding = next(ev for ev in events if ev[0] <= target < ev[1])
        window = [n for n in notes_for(rng, key, events) if target - 4 <= n["start_beat"] < target + 4]
        label = sounding[3]
        if label == "Other":
            answer = "a pitch-class set not in this list"
        elif label == "NoChord":
            answer = "no chord is sounding"
            window = [n for n in window if not (n["start_beat"] <= target < n["end_beat"])]
        else:
            answer = option_text(label)
        if rng.random() < .5:  # full vocabulary
            opts = [option_text(p) for p in CHORD_SETS] + ["no chord is sounding", "a pitch-class set not in this list"]
            ordered = True
        else:  # a short list containing the answer and near neighbours
            near = [option_text(p) for p in rng.sample(CHORD_SETS, 15)]
            opts = list(dict.fromkeys([answer] + near + ["no chord is sounding", "a pitch-class set not in this list"]))
            ordered = False
        if answer not in opts:
            continue
        shift = rng.choice([0, 0, 4, 16, 64])
        state = {"target_beat": target + shift,
                 "notes": [{**n, "start_beat": n["start_beat"] + shift, "end_beat": n["end_beat"] + shift} for n in window]}
        instr = rng.choice([
            "Which chord is sounding at the target beat? Pitch classes use C=0 ... B=11; enharmonic/equivalent names share an option.",
            "From the notes, identify the pitch-class set of the chord heard at the target beat (C=0 to B=11).",
            "Name the harmony at the target beat by its pitch-class set; ignore passing tones."])
        item = digest(seed, "chord", i)[:16]
        rows.append(row(source, "symbolic-chord", item, state, instr, opts, answer, item, ordered=ordered,
                        extra={"chord_label": label if isinstance(label, str) else list(label)}))
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=int, default=3000)
    parser.add_argument("--chords", type=int, default=12000)
    a = parser.parse_args()
    write(ROOT / "home-policy-train.jsonl", home_rows(a.home))
    write(ROOT / "symbolic-chord-train.jsonl", chord_rows(a.chords))
    print("written", ROOT)
