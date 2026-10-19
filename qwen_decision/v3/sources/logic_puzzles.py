"""Procedural reasoning puzzles in five BBH-like families, generated here with our own wording.

Upstream: none; every item is generated below from a fixed seed. Licence: ours. No BIG-Bench or
BBH task file is read (BBH, index catalog 58, is 23 BIG-Bench tasks; the plan bans their files),
and the scenarios, names and phrasings are our own rather than BBH's.

Families (1,000 items each, 5,000 in all):
  boolean    an expression of 4-7 truth values joined by and/or/not, in Python, upper-case word or
             logic-symbol (1/0, ∧ ∨ ¬) notation; answered as a True/False choice or as yes/no with
             either polarity.
  dates      day/week/year offsets from a stated date (in the answer format or spelled out), answered
             in one of four numeric date formats (gold plus 3-5
             near misses: off by a day or a week, wrong month or year, the offset applied backwards),
             or the weekday reached after n days given the start weekday.
  tracking   3-7 people each hold one distinct object and make 3-8 pairwise swaps; ask what one
             person ends with, or who ends with one object.
  sorting    which word lands in position k in alphabetical order, or which of four listings is
             alphabetical (the wrong ones are near misses).
  navigate   walking instructions: back at the start? (yes/no, balanced, absolute directions), where
             do you end relative to the start, or which way are you facing (turn-relative moves).
Overlap: the filter checks every row against the index (including the BBH rows) and JEVal. It masks
digits, so a bare date option such as "January 5, 2021" becomes "january 0 0" (or "jan 0 0"),
which some index rows hold as a short field; options are therefore numeric dates, and the symbol
notation writes 1/0 rather than T/F (strings of t/f letters also occur as short index fields).
"""
from __future__ import annotations

import datetime as dt
import random

from qwen_decision.v3.sources import common

NAME = "logic-puzzles-gen"
SEED = 5101
PER_FAMILY = 1000
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"}


def wrap(rng, text: str) -> str | dict:
    return text if rng.random() < .6 else {rng.choice(["puzzle", "problem", "task"]): text}


# ---------------------------------------------------------------- boolean
BOOL_CHOICE = [
    "What truth value does the expression in the state evaluate to?",
    "Evaluate the logical expression and pick its result.",
    "Work out the expression step by step; which value does it reduce to?",
]
BOOL_TRUE = ["Does the expression evaluate to true?", "The expression in the state is true."]
BOOL_FALSE = ["Does the expression come out false?", "The logical expression in the state evaluates to false."]
NOTATIONS = {
    "python": {"and": "and", "or": "or", "not": "not ", True: "True", False: "False"},
    "words": {"and": "AND", "or": "OR", "not": "NOT ", True: "TRUE", False: "FALSE"},
    "symbols": {"and": "∧", "or": "∨", "not": "¬", True: "1", False: "0"},
}


def bool_tree(rng, leaves: int):
    if leaves == 1:
        node = ("lit", rng.random() < .5)
    else:
        left = rng.randint(1, leaves - 1)
        node = (rng.choice(["and", "or"]), bool_tree(rng, left), bool_tree(rng, leaves - left))
    return ("not", node) if rng.random() < .3 else node


def bool_eval(node) -> bool:
    if node[0] == "lit":
        return node[1]
    if node[0] == "not":
        return not bool_eval(node[1])
    a, b = bool_eval(node[1]), bool_eval(node[2])
    return (a and b) if node[0] == "and" else (a or b)


def bool_text(node, sym, top=True) -> str:
    if node[0] == "lit":
        return sym[node[1]]
    if node[0] == "not":
        inner = bool_text(node[1], sym, False)
        return sym["not"] + (inner if node[1][0] in ("lit", "not") else f"({inner})" if inner[0] != "(" else inner)
    text = f"{bool_text(node[1], sym, False)} {sym[node[0]]} {bool_text(node[2], sym, False)}"
    return text if top else f"({text})"


def boolean(rng, i):
    want = i % 2 == 0
    while True:
        node = bool_tree(rng, rng.randint(4, 7))
        if bool_eval(node) == want:
            break
    notation = rng.choices(list(NOTATIONS), weights=[.6, .2, .2])[0]
    sym = NOTATIONS[notation]
    expr = bool_text(node, sym)
    state = expr if rng.random() < .5 else {"expression": expr, "notation": notation}
    if rng.random() < .5:
        options = [sym[True], sym[False]]
        rng.shuffle(options)
        q, target = common.choice(state, rng.choice(BOOL_CHOICE), options, options.index(sym[want]))
    elif rng.random() < .5:
        q, target = common.noul(rng.choice(BOOL_TRUE), want)
    else:
        q, target = common.noul(rng.choice(BOOL_FALSE), not want)
    return state, q, target


# ---------------------------------------------------------------- dates
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FORMATS = {
    "MM/DD/YYYY": lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}",
    "YYYY-MM-DD": lambda d: d.isoformat(),
    "DD.MM.YYYY": lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
    "YYYY/MM/DD": lambda d: f"{d.year}/{d.month:02d}/{d.day:02d}",
}
DATE_STORIES = [
    ("A library book was borrowed on {d}. It is due back {n} days later.", "days", 1, "On what date is it due?"),
    ("A parcel arrived on {d}, exactly {n} days after it was posted.", "days", -1, "On what date was it posted?"),
    ("A training course began on {d} and lasted {n} weeks.", "weeks", 1, "On what date did it end?"),
    ("The calendar says it is {d} today.", "days", -1, "What was the date {n} days ago?"),
    ("A warranty bought on {d} runs for exactly {n} years.", "years", 1, "On what date does it run out?"),
    ("A seed was planted on {d}; it sprouted {n} days later.", "days", 1, "On what date did it sprout?"),
    ("The concert is on {d}. Tickets went on sale {n} weeks earlier.", "weeks", -1, "When did ticket sales open?"),
]
DATE_CHOICE = [
    "Which date answers the question? Dates are written as {fmt}.",
    "Pick the correct date (format {fmt}).",
    "Work out the calendar arithmetic and select the right date, given in {fmt} form.",
]
WEEKDAY_CHOICE = ["Which day of the week is it then?", "Pick the weekday that results.",
                  "Count forward on the calendar and choose the right weekday."]


def shift(d: dt.date, unit: str, n: int) -> dt.date:
    if unit == "days":
        return d + dt.timedelta(days=n)
    if unit == "weeks":
        return d + dt.timedelta(weeks=n)
    return d.replace(year=d.year + n)


def near_misses(rng, gold: dt.date, start: dt.date, unit: str, n: int, sign: int) -> list[dt.date]:
    out = [gold + dt.timedelta(days=1), gold - dt.timedelta(days=1), gold + dt.timedelta(days=7),
           gold - dt.timedelta(days=7), shift(start, unit, -sign * n)]
    other = []
    for months in (1, -1):
        m = gold.month + months
        other.append(dict(year=gold.year + (m - 1) // 12, month=(m - 1) % 12 + 1))
    other.append(dict(year=gold.year + rng.choice([1, -1])))
    if gold.day <= 12 and gold.day != gold.month:
        other.append(dict(month=gold.day, day=gold.month))
    for change in other:
        try:
            out.append(gold.replace(**change))
        except ValueError:  # 29-31 in a shorter month, or 29 February in a common year
            continue
    return out


def dates(rng, i):
    if rng.random() < .2:
        start_day = rng.randrange(7)
        n = rng.randint(3, 120)
        text = f"{rng.choice(['A festival opens', 'A ship sails', 'A trial starts'])} on a {WEEKDAYS[start_day]}. " \
               f"{rng.choice(['The next event', 'The follow-up', 'The review'])} is {n} days later."
        q, target = common.choice(text, rng.choice(WEEKDAY_CHOICE), WEEKDAYS, (start_day + n) % 7)
        return wrap(rng, text), q, target
    while True:
        start = dt.date(1900, 1, 1) + dt.timedelta(days=rng.randrange(73000))
        if start.day <= 28:
            break
    story, unit, sign, ask = rng.choice(DATE_STORIES)
    n = {"days": rng.randint(2, 400), "weeks": rng.randint(1, 20), "years": rng.randint(1, 30)}[unit]
    gold = shift(start, unit, sign * n)
    fmt = rng.choice(list(FORMATS))
    show = FORMATS[fmt]
    misses = list(dict.fromkeys(m for m in near_misses(rng, gold, start, unit, n, sign) if m != gold))
    rng.shuffle(misses)
    options = [show(gold)] + [show(m) for m in misses[:rng.randint(3, 5)]]
    options = list(dict.fromkeys(options))
    rng.shuffle(options)
    told = show(start) if rng.random() < .5 else f"{MONTHS[start.month - 1]} {start.day}, {start.year}"
    text = story.format(d=told, n=n) + " " + ask.format(n=n)
    q, target = common.choice(text, rng.choice(DATE_CHOICE).format(fmt=fmt), options, options.index(show(gold)))
    return wrap(rng, text), q, target


# ---------------------------------------------------------------- tracking
PEOPLE = ["Priya", "Tomasz", "Keiko", "Mateo", "Ngozi", "Lars", "Farah", "Diego", "Ines", "Kofi", "Yuki", "Omar",
          "Svetlana", "Ravi", "Amara", "Bjorn", "Chiara", "Dmitri", "Esme", "Hamid", "Leilani", "Nikos"]
THINGS = [
    ("marble", ["red marble", "blue marble", "green marble", "amber marble", "white marble", "black marble",
                "violet marble"], "holds a {x}", "swap marbles"),
    ("book", ["atlas", "cookbook", "poetry collection", "field guide", "comic book", "dictionary", "biography"],
     "is reading a {x}", "trade books"),
    ("seat", ["seat 1", "seat 2", "seat 3", "seat 4", "seat 5", "seat 6", "seat 7"], "sits in {x}",
     "switch seats"),
    ("instrument", ["cello", "flute", "drum", "harp", "trumpet", "banjo", "oboe"], "plays the {x}",
     "exchange instruments"),
    ("hat", ["beret", "fedora", "sun hat", "beanie", "top hat", "cap", "cowboy hat"], "wears a {x}",
     "swap hats"),
]
TRACK_HOLDER = ["After all the exchanges, what does {p} end up with?", "Which {k} does {p} have at the end?",
                "Follow every swap: what is {p}'s {k} once they are done?"]
TRACK_OWNER = ["Who ends up with the {x}?", "After the exchanges, which person has the {x}?",
               "Track the swaps: who holds the {x} at the end?"]


def tracking(rng, i):
    n = rng.randint(3, 7)
    people = rng.sample(PEOPLE, n)
    kind, things, has, verb = rng.choice(THINGS)
    things = rng.sample(things, n)
    owner = dict(zip(people, things))
    lines = [f"{len(people)} friends each start with one {kind}: " +
             "; ".join(f"{p} {has.format(x=t)}" for p, t in owner.items()) + "."]
    for step in range(rng.randint(3, 8)):
        a, b = rng.sample(people, 2)
        owner[a], owner[b] = owner[b], owner[a]
        lead = rng.choice(["Then", "Next", "After that", "Later"]) if step else "First"
        lines.append(f"{lead}, {a} and {b} {verb}.")
    text = " ".join(lines)
    if rng.random() < .6:
        p = rng.choice(people)
        options = list(things)
        q, target = common.choice(text, rng.choice(TRACK_HOLDER).format(p=p, k=kind), options,
                                  options.index(owner[p]))
    else:
        x = rng.choice(things)
        options = list(people)
        holder = next(p for p, t in owner.items() if t == x)
        q, target = common.choice(text, rng.choice(TRACK_OWNER).format(x=x), options, options.index(holder))
    return wrap(rng, text), q, target


# ---------------------------------------------------------------- sorting
VOCAB = ("cart carton cartel carpet carrot carbon candle canopy cable cabin cobalt cobble comet comic compass "
         "paddle padlock pallet palette panel pantry parcel parrot pasta pastel patrol pebble pelican pencil "
         "pepper permit sable saddle saffron salmon salute sandal satin sauce scarf scheme sketch skillet "
         "slate sleeve slogan meadow medal melon mentor merit metal mirror mitten mosaic motel muffin mural "
         "tablet tackle talon tango tavern teapot temple tender thimble thistle ticket timber tinsel toffee "
         "bacon badge bagel ballad bamboo banner barrel basket beacon beaver bellow beret biscuit blanket "
         "garden garlic garnet gazebo gecko ginger glacier goblet gravel griffin grotto guitar harbor harness "
         "hazel helmet hermit hollow honey hornet zealot zenith zephyr zigzag violin velvet vessel vigil").split()
SORT_POSITION = ["Sort the words alphabetically. Which word is {o}?",
                 "In dictionary order, which of these words would come {o}?",
                 "Arrange the list from A to Z; which word lands in position {k}?"]
SORT_LISTING = ["Which line gives the words in correct alphabetical order?",
                "Pick the listing that is sorted from A to Z.",
                "Which ordering of the words is alphabetical?"]


def sorting(rng, i):
    words = rng.sample(VOCAB, rng.randint(5, 9))
    ordered = sorted(words)
    if rng.random() < .6:
        k = rng.randint(1, len(words))
        if rng.random() < .3:
            k = len(words)
            o = "last"
        else:
            o = ORD[k]
        text = "Words: " + ", ".join(words)
        q, target = common.choice(text, rng.choice(SORT_POSITION).format(o=o, k=k), list(words),
                                  words.index(ordered[k - 1]))
        return wrap(rng, text), q, target
    wrong = []
    j = rng.randrange(len(ordered) - 1)
    swap = ordered[:j] + [ordered[j + 1], ordered[j]] + ordered[j + 2:]
    wrong += [swap, sorted(words, key=lambda w: w[::-1]), sorted(words, key=lambda w: (len(w), w)),
              ordered[::-1], sorted(words, key=lambda w: w[1:])]
    seen, options = {", ".join(ordered)}, [", ".join(ordered)]
    for w in wrong:
        line = ", ".join(w)
        if line not in seen:
            seen.add(line)
            options.append(line)
    options = [options[0]] + rng.sample(options[1:], min(3, len(options) - 1))
    rng.shuffle(options)
    text = "Words: " + ", ".join(rng.sample(words, len(words)))
    q, target = common.choice(text, rng.choice(SORT_LISTING), options, options.index(", ".join(ordered)))
    return wrap(rng, text), q, target


# ---------------------------------------------------------------- navigate
COMPASS = ["north", "east", "south", "west"]
STEP = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
NAV_BACK = ["After following every instruction, are you back where you began?",
            "Do these moves return you to the starting spot?"]
NAV_AWAY = ["After all the moves, are you somewhere other than the starting point?",
            "The instructions leave you away from where you started."]
NAV_WHERE = ["Where do you end up relative to the starting point?",
             "Which option describes your final position compared with where you started?",
             "Follow the moves; where are you relative to the start?"]
NAV_FACING = ["Which direction are you facing at the end?", "After the last instruction, which way do you face?"]


def describe(dx: int, dy: int) -> str:
    parts = []
    if dy:
        parts.append(f"{abs(dy)} {'pace' if abs(dy) == 1 else 'paces'} {'north' if dy > 0 else 'south'}")
    if dx:
        parts.append(f"{abs(dx)} {'pace' if abs(dx) == 1 else 'paces'} {'east' if dx > 0 else 'west'}")
    return " and ".join(parts) + " of the start" if parts else "exactly at the start"


def navigate(rng, i):
    if rng.random() < .4:  # absolute directions, balanced yes/no
        moves = [(rng.choice(COMPASS), rng.randint(1, 9)) for _ in range(rng.randint(2, 5))]
        back = i % 2 == 0
        x = sum(STEP[d][0] * n for d, n in moves)
        y = sum(STEP[d][1] * n for d, n in moves)
        if back:
            for axis, value in (("east", x), ("north", y)):
                if value:
                    d = {"east": "west", "north": "south"}[axis] if value > 0 else axis
                    left = abs(value)
                    while left:
                        n = rng.randint(1, left)
                        moves.append((d, n))
                        left -= n
            rng.shuffle(moves)
        else:
            if x == 0 and y == 0:
                moves.append((rng.choice(COMPASS), rng.randint(1, 5)))
        x = sum(STEP[d][0] * n for d, n in moves)
        y = sum(STEP[d][1] * n for d, n in moves)
        text = " ".join(f"Walk {n} {'pace' if n == 1 else 'paces'} {d}." for d, n in moves)
        if rng.random() < .5:
            q, target = common.noul(rng.choice(NAV_BACK), x == 0 and y == 0)
        else:
            q, target = common.noul(rng.choice(NAV_AWAY), not (x == 0 and y == 0))
        return wrap(rng, text), q, target
    facing, x, y, lines = 0, 0, 0, ["You start facing north."]
    for _ in range(rng.randint(3, 8)):
        r = rng.random()
        if r < .35:
            turn = rng.choice([("Pivot a quarter turn to your left.", -1), ("Pivot a quarter turn to your right.", 1),
                               ("Spin round to face the opposite way.", 2)])
            facing = (facing + turn[1]) % 4
            lines.append(turn[0])
        else:
            n = rng.randint(1, 6)
            back = rng.random() < .25
            sx, sy = STEP[COMPASS[facing]]
            x, y = (x - sx * n, y - sy * n) if back else (x + sx * n, y + sy * n)
            paces = "pace" if n == 1 else "paces"
            lines.append(f"Back up {n} {paces}." if back else f"Move ahead {n} {paces}.")
    text = " ".join(lines)
    if rng.random() < .3:
        q, target = common.choice(text, rng.choice(NAV_FACING), list(COMPASS), facing)
        return wrap(rng, text), q, target
    gold = describe(x, y)
    alts = [describe(-x, y), describe(x, -y), describe(y, x), describe(-x, -y), describe(x + 1, y), describe(x, y - 1),
            describe(0, 0)]
    options = [gold] + [a for a in dict.fromkeys(alts) if a != gold]
    options = [gold] + rng.sample(options[1:], min(rng.randint(3, 5), len(options) - 1))
    rng.shuffle(options)
    q, target = common.choice(text, rng.choice(NAV_WHERE), options, options.index(gold))
    return wrap(rng, text), q, target


FAMILIES = {"boolean": boolean, "dates": dates, "tracking": tracking, "sorting": sorting, "navigate": navigate}


def build():
    for f, (family, make) in enumerate(FAMILIES.items()):
        rng = random.Random(SEED * 10 + f)
        for i in range(PER_FAMILY):
            state, q, target = make(rng, i)
            yield common.example(NAME, f"{family}:{i}", "logic", state, q, target, "generated")


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "generated (qwen_decision/v3/sources/logic_puzzles.py)",
                                 "revision": f"seed {SEED}", "split": "generated", "licence": "ours (generated)",
                                 "families": list(FAMILIES), "per_family": PER_FAMILY})
