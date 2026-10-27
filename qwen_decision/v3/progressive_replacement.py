"""Theseus-inspired schedule for the SAME previously tested BDH successor.

One stochastic training route replaces the upper eight native blocks with the
shared three-cycle core. At inference only the successor executes. Unlike the
original frozen-embedding Theseus recipe, the common prefix and head are fully
fine-tuned throughout, matching our existing abrupt-replacement control.
"""
import hashlib

from qwen_decision.v3 import bdh_memory as bdh, modeling


def replacement_probability(step, transition_steps=1024, initial=.1):
    if step < 1 or transition_steps < 2 or not 0 <= initial <= 1:
        raise ValueError("invalid replacement schedule")
    return min(1., initial + (1. - initial) * (step - 1) / (transition_steps - 1))


def use_successor(seed, step, transition_steps=1024, initial=.1):
    probability = replacement_probability(step, transition_steps, initial)
    # Independent of global RNG and checkpoint replay; one route for an update.
    value = int.from_bytes(hashlib.sha256(f"theseus-bdh-v1:{seed}:{step}".encode()).digest()[:8], "big")
    return value / 2**64 < probability


def logits(model, ids, mask, codes, valid, *, successor=True):
    if successor:
        return bdh.logits(model, ids, mask, codes, valid)
    # Frozen predecessor blocks still propagate gradients into the common prefix.
    return modeling.code_logits(model, ids, mask, codes, valid)
