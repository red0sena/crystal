"""Code output prediction: small generated Python functions, pick the call's return value from 2-10 candidates.

Upstream: none. Every function comes from our own templates below (30 families: list filters and
folds, string rewrites, counters, bounded loops, windows, pair counts, slicing, keyed sorts, digit
arithmetic, Caesar shifts, zips, early returns, stacks, recursion, min/max tracking, set algebra,
divmod and bit arithmetic, list mutation, padding, grids, run-length coding...), with names drawn
from our own pools and constants, operators and inputs drawn from a fixed seed. Licence: ours.

Gold: the call is executed at BUILD time only, on CPU, in a separate `python -I` process with a
per-call alarm, an address-space limit and PYTHONHASHSEED=0 (no set order reaches an output). The
model never sees the execution; it must trace the code. Distractors are real outputs of plausible
mistakes: the same call on mutants of the function (one operator, constant or slice changed), the
function on other inputs, and type-preserving slips of the true value (off by one, reversed,
case-swapped, last element dropped). Every option is a Python literal no longer than 80 characters
and no option equals the gold.

Overlap: CRUXEval (index catalog 43; all 800 problems are test) and its samples are not used or
read. The filter matches every short line exactly, so a generic line such as `for ch in word:` or
`if r % 2 == 0:` that also occurs in a CRUXEval row would drop an unrelated item; the name pools
leave out the names that produced such collisions in a trial build (ch, c, word, line, items, data,
r, q, res, tab, size, target, ...). The filter still checks every row.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
from string import Template
import subprocess
import sys

from qwen_decision.v3.sources import common

NAME = "code-output-gen"
SEED = 20261016
N = 5000
MAX_REPR = 80
PHRASINGS = [
    "What does the call return? Trace the code by hand.",
    "Without running it, work out what the function call evaluates to.",
    "Which candidate is the exact return value of the call shown in the state?",
    "Predict the value produced by executing the given call.",
    "Mentally execute the code: which option equals the value returned?",
    "Which Python value does this call produce?",
]

# Placeholders: $FN function name; $L* list args; $S* string args; $W* word-list args; $I* int args;
# $V* locals; $E* list elements; $C* characters; $J* indices. Lower-case $slots are mutable choices.
POOLS = {
    "FN": [f"{v}_{n}" for v in ("scan", "fold", "mix", "tally", "walk", "squash", "shape", "sift", "merge", "twist",
                                "trace", "pack", "crunch", "shift", "weave", "probe", "pick", "spin", "build", "churn")
           for n in ("values", "items", "word", "seq", "data", "line", "parts", "row", "tokens", "bits", "list")],
    "L": ["values", "seq", "row", "series", "bag", "pool", "entries", "xs", "ys", "marks", "batch", "column"],
    "S": ["phrase", "label", "msg", "token", "src", "raw", "chunk", "blurb", "caption", "motto"],
    "W": ["words", "names", "terms", "labels", "tags", "keys_in"],
    "I": ["limit", "stride", "span", "tier", "level", "quota", "width", "bound", "reach", "radix"],
    "V": ["acc", "total", "outp", "buf", "agg", "score", "memo", "store", "cur", "tmp", "pile", "best", "holder",
          "box", "stack_", "kept", "piece", "head", "rest", "low", "high", "cnt", "carry", "bucket", "ledger"],
    "E": ["x", "v", "u", "z", "el", "elem", "w_"],
    "C": ["letter", "sym", "glyph", "ltr", "chr_"],
    "J": ["j", "idx", "pos", "at", "ix", "jj"],
}
WORDS = ("apple river stone cloud paper tiger lemon mango rocket violin candle forest garden silver window "
         "orange planet bridge castle dragon falcon harbor island jungle kettle ladder magnet needle oyster "
         "pepper quartz saddle tunnel umbrella velvet walnut yogurt zebra anchor banjo cactus dolphin ember "
         "fossil glacier hammer igloo jasmine kayak lantern meadow nectar orbit pebble quiver raven sprout "
         "thistle urchin vortex willow pixel nimbus cobalt delta echo fable gizmo hazel ivory jolly karma").split()


def ints(rng, lo=-5, hi=20, n=(3, 8)):
    return [rng.randint(lo, hi) for _ in range(rng.randint(*n))]


def word(rng):
    return rng.choice(WORDS)


def phrase(rng, n=(3, 6)):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(*n)))


def letters(rng, alphabet="abc", n=(5, 10)):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(*n)))


T = []  # (code template, slots {name: options}, args(rng) -> tuple)


def template(code, slots, args):
    T.append((code.strip("\n"), slots, args))


template("""
def $FN($L1):
    return [$E1 $op $c1 for $E1 in $L1 if $E1 $cmp $c2]
""", {"op": ["+", "-", "*", "//", "%"], "c1": [1, 2, 3, 4, 5], "cmp": [">", "<", ">=", "<=", "!="], "c2": [-1, 0, 2, 3, 5, 7]},
    lambda r: (ints(r),))

template("""
def $FN($L1):
    $V1 = $c0
    for $E1 in $L1:
        if $E1 % $c1 == 0:
            $V1 $aug1 $E1
        else:
            $V1 $aug2 $c2
    return $V1
""", {"c0": [0, 1, 2, 3], "c1": [2, 3, 4], "aug1": ["+=", "-=", "*="], "aug2": ["+=", "-=", "*="], "c2": [1, 2, 3]},
    lambda r: (ints(r, -3, 9, (3, 6)),))

template("""
def $FN($S1):
    $V1 = ''
    for $J1, $C1 in enumerate($S1):
        if $J1 % $c1 == $c2:
            $V1 += $C1.upper()
        else:
            $V1 += $C1$tail
    return $V1
""", {"c1": [2, 3, 4], "c2": [0, 1], "tail": ["", " * 2", ".lower()"]}, lambda r: (word(r),))

template("""
def $FN($S1):
    $V1 = $S1.split()
    $V2 = [$E1 for $E1 in $V1 if len($E1) $cmp $c1]
    return '$sep'.join($V2$rev)
""", {"cmp": [">", "<", ">=", "<=", "=="], "c1": [4, 5, 6], "sep": ["-", " ", "", "_", ","], "rev": ["", "[::-1]", "[1:]"]},
    lambda r: (phrase(r),))

template("""
def $FN($S1):
    $V1 = {}
    for $C1 in $S1:
        $V1[$C1] = $V1.get($C1, $c0) + $c1
    return $ret
""", {"c0": [0, 1], "c1": [1, 2], "ret": ["$V1", "max($V1, key=$V1.get)", "sorted($V1.items())[:3]", "len($V1)",
                                         "[$C1 for $C1 in $V1 if $V1[$C1] > $c1]"]},
    lambda r: (letters(r, "abcde", (5, 9)),))

template("""
def $FN($I1):
    $V1 = 0
    while $I1 $cmp $c1 and $V1 < $c3:
        if $I1 % 2 == 0:
            $I1 = $I1 // 2
        else:
            $I1 = $I1 * $c2 + 1
        $V1 += 1
    return $ret
""", {"cmp": [">", ">="], "c1": [1, 2, 3], "c2": [3, 1, 5], "c3": [8, 12, 20], "ret": ["($V1, $I1)", "$V1", "$I1 + $V1"]},
    lambda r: (r.randint(2, 60),))

template("""
def $FN($L1, $I1):
    $V1 = None
    for $J1 in range(len($L1) - $I1 + $off):
        $V2 = sum($L1[$J1:$J1 + $I1])
        if $V1 is None or $V2 $cmp $V1:
            $V1 = $V2
    return $V1
""", {"off": [1, 0], "cmp": [">", "<", ">=", "<="]}, lambda r: (ints(r, -6, 12, (4, 8)), r.randint(2, 3)))

template("""
def $FN($S1):
    return $S1.replace('$a', '$b').split('$c')$tail
""", {"a": ["a", "e", "o", "r", "n"], "b": ["", "x", "aa", " ", "-"], "c": [" ", "a", "e", "o"], "tail": ["", "[1:]", "[::-1]", "[0]", "[-1]"]},
    lambda r: (phrase(r, (2, 4)),))

template("""
def $FN($L1, $I1):
    $V1 = 0
    for $J1 in range(len($L1)):
        for $J2 in range($J1 + 1, len($L1)):
            if $L1[$J1] $op $L1[$J2] $cmp $I1:
                $V1 += 1
    return $V1
""", {"op": ["+", "-", "*"], "cmp": ["==", ">", "<", ">="]}, lambda r: (ints(r, -3, 9, (4, 7)), r.randint(0, 10)))

template("""
def $FN($L1):
    $V1 = $L1[$s1:$s2]
    $V2 = $L1[$s3::$s4]
    return $V1 + $V2
""", {"s1": [0, 1, 2], "s2": [3, 4, 5, -1], "s3": [-3, -2, 0, 1], "s4": [1, -1, 2]}, lambda r: (ints(r, 0, 30, (5, 8)),))

template("""
def $FN($W1):
    return sorted($W1, key=lambda $E1: $key, reverse=$rv)
""", {"key": ["(len($E1), $E1)", "$E1[-1]", "(-len($E1), $E1)", "($E1.count('a'), $E1)", "$E1[1:]"], "rv": ["True", "False"]},
    lambda r: (r.sample(WORDS, r.randint(3, 5)),))

template("""
def $FN($S1, $S2):
    return $ret
""", {"ret": ["($S1.find($S2), $S1.count($S2))", "$S1.rfind($S2) - $S1.find($S2)",
              "[$J1 for $J1, $C1 in enumerate($S1) if $C1 == $S2]", "($S1.find($S2), $S1.rfind($S2))",
              "$S1.count($S2) * len($S1)"]},
    lambda r: (phrase(r, (1, 3)), r.choice("aeiorstn")))

template("""
def $FN($I1):
    $V1 = $c0
    for $C1 in str($I1):
        $V1 = $V1 $op int($C1)$extra
    return $V1
""", {"c0": [0, 1, 2], "op": ["+", "*", "-"], "extra": ["", " + 1", " * 2"]}, lambda r: (r.randint(10, 99999),))

template("""
def $FN($S1, $I1):
    $V1 = []
    for $C1 in $S1:
        if $C1.isalpha():
            $V1.append(chr((ord($C1) - 97 $sign $I1) % 26 + 97))
        else:
            $V1.append($C1)
    return ''.join($V1)$tail
""", {"sign": ["+", "-"], "tail": ["", "[::-1]", ".upper()"]}, lambda r: (phrase(r, (1, 2)), r.randint(1, 5)))

template("""
def $FN($L1, $L2):
    return [$E1 $op $E2 for $E1, $E2 in zip($L1$sa, $L2$sb)]
""", {"op": ["+", "-", "*", "%", "//"], "sa": ["", "[1:]", "[::-1]"], "sb": ["", "[1:]", "[::-1]"]},
    lambda r: (ints(r, 1, 12, (3, 6)), ints(r, 1, 9, (3, 6))))

template("""
def $FN($L1, $I1):
    for $J1, $E1 in enumerate($L1):
        if $E1 $cmp $I1:
            return $J1$adj
    return $miss
""", {"cmp": [">", "<", "==", ">="], "adj": ["", " * 2", " + 1", " - 1"], "miss": ["-1", "len($L1)", "None"]},
    lambda r: (ints(r, -4, 15, (4, 8)), r.randint(0, 12)))

template("""
def $FN($S1):
    $V1 = []
    for $C1 in $S1:
        if $V1 and $V1[-1] $cmp $C1:
            $V1.pop()
        else:
            $V1.append($C1)
    return ''.join($V1)$tail
""", {"cmp": ["==", "!=", "<", ">"], "tail": ["", "[::-1]", ".upper()"]}, lambda r: (letters(r, "abc", (5, 10)),))

template("""
def $FN($I1):
    if $I1 $base:
        return $bv
    return $FN($I1 - 1) $op $FN($I1 - 2)$extra
""", {"base": ["<= 1", "< 2", "<= 0", "< 3"], "bv": ["$I1", "1", "2"], "op": ["+", "-", "*"], "extra": ["", " + 1"]},
    lambda r: (r.randint(3, 9),))

template("""
def $FN($L1):
    $V1, $V2 = $L1[0], $L1[0]
    for $E1 in $L1[1:]:
        if $E1 $cmp1 $V1:
            $V1 = $E1
        if $E1 $cmp2 $V2:
            $V2 = $E1
    return $ret
""", {"cmp1": ["<", ">", "<="], "cmp2": [">", "<", ">="], "ret": ["($V1, $V2)", "$V2 - $V1", "[$V1, $V2, $V2 - $V1]"]},
    lambda r: (ints(r, -9, 20, (3, 7)),))

template("""
def $FN($S1):
    return '$sep'.join(f'{$C1}{$J1$o}' for $J1, $C1 in enumerate($S1) if $cond)
""", {"sep": ["", "-", ","], "o": ["", " + 1", " * 2"], "cond": ["$C1 in 'aeiou'", "$C1 not in 'aeiou'", "$C1 > 'm'", "$C1 < 'h'"]},
    lambda r: (word(r),))

template("""
def $FN($L1, $L2):
    return sorted(set($L1) $op set($L2))$tail
""", {"op": ["&", "|", "-", "^"], "tail": ["", "[::-1]", "[:3]", "[-2:]"]},
    lambda r: (ints(r, 0, 9, (3, 7)), ints(r, 0, 9, (3, 7))))

template("""
def $FN($I1, $I2):
    $V1, $V2 = divmod($I1, $I2)
    return $V1 $op1 $V2 $op2 $c1
""", {"op1": ["+", "-", "*"], "op2": ["+", "-", "*"], "c1": [1, 2, 3]}, lambda r: (r.randint(10, 99), r.randint(2, 9)))

template("""
def $FN($I1, $I2):
    return ($I1 $bop $I2) $op ($I1 $sh $c1)
""", {"bop": ["&", "|", "^"], "op": ["+", "-", "^"], "sh": ["<<", ">>"], "c1": [1, 2, 3]},
    lambda r: (r.randint(1, 63), r.randint(1, 63)))

template("""
def $FN($L1):
    $L1 = list($L1)
    $L1.append($c1)
    $L1.insert($c2, $L1.pop($c3))
    $L1.$m1
    return $L1
""", {"c1": [0, 5, 9], "c2": [0, 1, 2, 3], "c3": [-1, 0, 1], "m1": ["reverse()", "sort()", "remove($L1[0])", "extend($L1[:2])"]},
    lambda r: (ints(r, 0, 20, (3, 6)),))

template("""
def $FN($S1, $I1):
    return $S1.$pad($I1, '$fill')$tail
""", {"pad": ["center", "ljust", "rjust"], "fill": ["*", "-", "."], "tail": ["", ".upper()", "[::2]", "[::-1]"]},
    lambda r: (word(r), r.randint(6, 11)))

template("""
def $FN($W1):
    $V1 = {}
    for $E1 in $W1:
        $V2 = $key
        $V1[$V2] = $V1.get($V2, $c0) + $inc
    return sorted($V1.items())$tail
""", {"key": ["$E1[0]", "len($E1)", "$E1[-1]"], "c0": [0, 1], "inc": ["1", "len($E1)"], "tail": ["", "[:3]", "[-2:]"]},
    lambda r: (r.sample(WORDS, r.randint(3, 6)),))

template("""
def $FN($L1):
    return ($q1($E1 $cmp $c1 for $E1 in $L1), $q2($E1 % $c2 == 0 for $E1 in $L1))
""", {"q1": ["any", "all"], "q2": ["any", "all"], "cmp": [">", "<", ">="], "c1": [0, 3, 5, 10], "c2": [2, 3, 5]},
    lambda r: (ints(r, -3, 15, (3, 6)),))

template("""
def $FN($S1):
    $V1 = []
    $V2, $V3 = $S1[0], $c0
    for $C1 in $S1[1:]:
        if $C1 == $V2:
            $V3 += 1
        else:
            $V1.append($fmt)
            $V2, $V3 = $C1, $c0
    $V1.append($fmt)
    return '$sep'.join($V1)
""", {"c0": [1, 0], "fmt": ["f'{$V2}{$V3}'", "f'{$V3}{$V2}'", "$V2 * $V3"], "sep": ["", "-", "|"]},
    lambda r: (letters(r, "abc", (5, 10)),))

template("""
def $FN($I1):
    $V1 = [[$cell for $J2 in range($I1)] for $J1 in range($I1)]
    return $ret
""", {"cell": ["$J1 * $J2", "$J1 + $J2", "($J1 + $J2) % 2", "$J1 - $J2"],
      "ret": ["[$V2[-1] for $V2 in $V1]", "sum(map(sum, $V1))", "$V1[$I1 - 1]", "[$V1[$J1][$J1] for $J1 in range($I1)]"]},
    lambda r: (r.randint(2, 5),))

template("""
def $FN($S1):
    $S1 = $S1.$m1()
    $S1 = $S1.replace(' ', '$sep')
    return $S1$tail
""", {"m1": ["strip", "title", "upper", "swapcase", "capitalize"], "sep": ["", "_", "-", "."],
      "tail": ["", "[::-1]", "[:7]", ".count('a')", "[2:]"]},
    lambda r: (r.choice(["", " "]) + phrase(r, (2, 4)) + r.choice(["", "  "]),))

template("""
def $FN($L1):
    $L1 = list($L1)
    $V1 = []
    while $L1:
        $V1.append($L1.pop($idx))
        if len($V1) $cmp $c1:
            break
    return $V1
""", {"idx": ["0", "-1", "len($L1) // 2"], "cmp": [">=", ">", "=="], "c1": [2, 3, 4]}, lambda r: (ints(r, 0, 20, (3, 7)),))


def render(code: str, slots: dict, names: dict) -> str:
    text = Template(Template(code).safe_substitute(slots)).safe_substitute(slots)  # slot values may name slots
    return Template(text).substitute(names)


def placeholders(code: str, slots: dict) -> list[str]:
    text = code + " " + " ".join(str(o) for opts in slots.values() for o in opts)
    return sorted(set(re.findall(r"\$([A-Z]+[0-9]*)", text)))


def assign_names(rng: random.Random, keys: list[str]) -> dict:
    names, used = {}, set()
    for key in keys:
        pool = [n for n in POOLS[key.rstrip("0123456789")] if n not in used]
        names[key] = rng.choice(pool)
        used.add(names[key])
    return names


def call_text(names: dict, args: tuple) -> str:
    return f"{names['FN']}({', '.join(repr(a) for a in args)})"


RUNNER = r"""
import ast, json, resource, signal, sys
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
class Slow(Exception):
    pass
def alarm(*_):
    raise Slow()
signal.signal(signal.SIGALRM, alarm)
sys.setrecursionlimit(500)
out = []
for code, call in json.load(sys.stdin):
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        ns = {}
        exec(code, ns)
        value = eval(call, ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        text = repr(value)
        if len(text) > 200 or ast.literal_eval(text) != value:
            text = None
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        text = None
    out.append(text)
json.dump(out, sys.stdout)
"""


def execute(jobs: list[tuple[str, str]], chunk: int = 1500, workers: int = 8) -> list[str | None]:
    """repr of each (code, call) result, or None for errors, timeouts and non-literal values."""
    env = {"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": ""}

    def run(part):
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps(part), capture_output=True,
                               text=True, timeout=600, env=env)
            return json.loads(p.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return [None] * len(part)

    parts = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    with ThreadPoolExecutor(workers) as pool:
        return [x for res in pool.map(run, parts) for x in res]


def slips(value) -> list:
    """Type-preserving slips of the true value."""
    out = []
    if isinstance(value, bool):
        out.append(not value)
    elif isinstance(value, int):
        out += [value + 1, value - 1, -value, value * 2]
    elif isinstance(value, str):
        out += [value[::-1], value.swapcase(), value[:-1], value + value[-1:]]
    elif isinstance(value, (list, tuple)) and value:
        out += [type(value)(reversed(value)), type(value)(value[:-1])]
        if len(value) > 1:
            out.append(type(value)([value[1], value[0], *value[2:]]))
    elif isinstance(value, dict) and value:
        out.append(dict(list(value.items())[:-1]))
    return out


def candidates(rng: random.Random, count: int):
    """Generated items: code, names, args, and the (code, call) jobs whose outputs become options."""
    items = []
    seen = set()
    while len(items) < count:
        t = rng.randrange(len(T))
        code, slots, make_args = T[t]
        choice = {k: rng.choice(v) for k, v in slots.items()}
        names = assign_names(rng, placeholders(code, slots))
        source = render(code, choice, names)
        args = make_args(rng)
        call = call_text(names, args)
        if (source, call) in seen:
            continue
        seen.add((source, call))
        jobs = [(source, call)]
        mutants = set()
        for _ in range(12):
            k = rng.choice(list(slots))
            alt = [o for o in slots[k] if o != choice[k]]
            if not alt:
                continue
            m = render(code, {**choice, k: rng.choice(alt)}, names)
            if m != source and m not in mutants:
                mutants.add(m)
                jobs.append((m, call))
            if len(mutants) == 6:
                break
        for _ in range(3):
            jobs.append((source, call_text(names, make_args(rng))))
        items.append({"template": t, "code": source, "call": call, "jobs": jobs})
    return items


def build():
    rng = random.Random(SEED)
    items = candidates(rng, int(N * 1.4))
    flat = [job for it in items for job in it["jobs"]]
    results = execute(flat)
    at = 0
    kept = 0
    for i, it in enumerate(items):
        outs = results[at:at + len(it["jobs"])]
        at += len(it["jobs"])
        gold = outs[0]
        if gold is None or len(gold) > MAX_REPR:
            continue
        value = ast.literal_eval(gold)
        pool, seen = [], {gold}
        mutant_outs = [o for o in outs[1:-3] if o]
        input_outs = [o for o in outs[-3:] if o]
        rng.shuffle(mutant_outs)
        for o in mutant_outs + input_outs + [repr(s) for s in slips(value)]:
            if o in seen or len(o) > MAX_REPR:
                continue
            try:
                if ast.literal_eval(o) == value:
                    continue
            except (ValueError, SyntaxError):
                continue
            seen.add(o)
            pool.append(o)
        if not pool or (not mutant_outs and not input_outs):
            continue
        k = min(rng.randint(2, 10), len(pool) + 1)
        options = pool[:k - 1] + [gold]
        rng.shuffle(options)
        style = rng.random()
        if style < .35:
            state = {"code": it["code"], "call": it["call"]}
        elif style < .6:
            state = {"function": it["code"], "input": it["call"]}
        elif style < .8:
            state = f"```python\n{it['code']}\n```\n\nCall: {it['call']}"
        else:
            state = f"{it['code']}\n\n# evaluated expression\n{it['call']}"
        question, target = common.choice(state, common.instruction(rng, PHRASINGS), options, options.index(gold))
        yield common.example(NAME, f"{i}:t{it['template']}", "code-output", state, question, target, "generated")
        kept += 1
        if kept == N:
            return


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "generated (qwen_decision/v3/sources/code_output.py templates)",
                                 "revision": f"seed {SEED}", "split": "generated", "licence": "ours (generated)",
                                 "cast": "choice over 2-10 Python literals: true output, outputs of mutated functions "
                                         "and of other inputs, and slips of the true value",
                                 "templates": len(T)})
