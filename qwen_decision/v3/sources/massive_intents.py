"""MASSIVE 1.1 en-US train (11,514 utterances) as a 60-option voice-assistant intent choice.

Upstream: AmazonScience/massive (CC-BY-4.0). The Hub repo is a loading script over an S3 tarball, so
the rows come from the Hub's own parquet conversion of that script (refs/convert/parquet, pinned
commit below; the script repo itself is pinned too). en-US train only.

Overlap: MASSIVE is built on SLURP, which reuses HWU64 text, and JEVal nlu-eval-68 samples from the
whole of HWU64; the plan measured 98 of its 200 utterances verbatim in MASSIVE en-US (55 in train).
The contamination filter drops them (exact normalized short-text match). Not an index benchmark.
Cap 12,000 (all of train fits).

Options are plain-English descriptions of the 60 intents (scenario_intent codes are not readable).
"""
from __future__ import annotations

import json
import random

import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

NAME = "massive-intents"
REPO = "AmazonScience/massive"
PARQUET_REVISION = "ed58ac423a2f4121720918bf5301577edce4ffd3"  # refs/convert/parquet
SCRIPT_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
CAP = 12_000
READABLE = {
    "datetime_query": "ask for the date or time", "datetime_convert": "convert between time zones or formats",
    "iot_hue_lightchange": "change the colour of the lights", "iot_hue_lightup": "make the lights brighter",
    "iot_hue_lightdim": "dim the lights", "iot_hue_lightoff": "turn the lights off", "iot_hue_lighton": "turn the lights on",
    "iot_wemo_on": "switch a smart plug on", "iot_wemo_off": "switch a smart plug off",
    "iot_cleaning": "start the robot vacuum or cleaner", "iot_coffee": "make a coffee",
    "transport_ticket": "book a train or travel ticket", "transport_traffic": "check traffic conditions",
    "transport_query": "ask about transport, routes or timetables", "transport_taxi": "book a taxi",
    "takeaway_query": "ask about a takeaway order or delivery", "takeaway_order": "order takeaway food",
    "qa_stock": "ask about a stock price", "qa_currency": "ask about currency exchange rates",
    "qa_definition": "ask for the definition of a word", "qa_maths": "ask a maths question", "qa_factoid": "ask a general-knowledge question",
    "general_greet": "greet the assistant", "general_joke": "ask for a joke", "general_quirky": "chit-chat or an odd question",
    "recommendation_events": "ask for event recommendations", "recommendation_locations": "ask for places to go nearby",
    "recommendation_movies": "ask for a movie recommendation",
    "music_dislikeness": "say they dislike a song or artist", "music_likeness": "say they like a song or artist",
    "music_settings": "change music playback settings (shuffle, repeat)", "music_query": "ask about the music that is playing",
    "play_music": "play music", "play_radio": "play a radio station", "play_audiobook": "play an audiobook",
    "play_podcasts": "play a podcast", "play_game": "play a game",
    "cooking_recipe": "ask for a recipe", "cooking_query": "ask a cooking question",
    "weather_query": "ask about the weather", "news_query": "ask for news",
    "audio_volume_up": "turn the volume up", "audio_volume_down": "turn the volume down",
    "audio_volume_mute": "mute the sound", "audio_volume_other": "other volume or audio setting",
    "email_addcontact": "add an email contact", "email_querycontact": "look up a contact's details",
    "email_sendemail": "send an email", "email_query": "check emails",
    "lists_createoradd": "create a list or add an item to one", "lists_remove": "remove a list or a list item",
    "lists_query": "ask what is on a list",
    "alarm_set": "set an alarm", "alarm_query": "check which alarms are set", "alarm_remove": "remove an alarm",
    "calendar_set": "add a calendar event or reminder", "calendar_query": "ask about calendar events",
    "calendar_remove": "delete a calendar event",
    "social_query": "check social media updates", "social_post": "post on social media or complain to a company",
}
PHRASINGS = [
    "A user said this to a voice assistant. What do they want it to do?",
    "Which of these assistant intents matches the spoken command?",
    "Pick the intent a smart-speaker assistant should recognise in this utterance.",
    "Classify the user's spoken request into one of the assistant's 60 intents.",
    "What is the user asking the home assistant for?",
]
STATE_KEYS = ["utterance", "spoken_command", "user_said", "transcript"]


def build():
    local, revision = common.fetch(REPO, ["en-US/train/*"], revision=PARQUET_REVISION)
    common.fetch(REPO, ["README.md", "LICENSE"], revision=SCRIPT_REVISION)
    table = pq.read_table(local / "en-US/train/0000.parquet")
    codes = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    assert len(codes) == 60 and set(codes) == set(READABLE)
    labels = [READABLE[c] for c in codes]
    rows = table.select(["id", "partition", "intent", "utt"]).to_pylist()
    assert all(r["partition"] == "train" for r in rows)
    rng = random.Random(60)
    rng.shuffle(rows)
    for r in rows[:CAP]:
        options = sorted(labels) if rng.random() < .5 else rng.sample(labels, len(labels))
        state = r["utt"] if rng.random() < .3 else {rng.choice(STATE_KEYS): r["utt"]}
        question, target = common.choice(state, common.instruction(rng, PHRASINGS), options, options.index(labels[r["intent"]]))
        yield common.example(NAME, f"en-US:train:{r['id']}", "intent", state, question, target, "train")


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": REPO, "revision": PARQUET_REVISION, "script_revision": SCRIPT_REVISION,
                                 "locale": "en-US", "split": "train", "cap": CAP, "licence": "CC-BY-4.0"})
