import csv

import pytest

from logfusion.evaluation import grouping_accuracy
from logfusion.parsing.masking import mask, tokenize
from logfusion.parsing.miner import WILDCARD, TemplateMiner


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
            "Receiving block <BLK> src: <IP> dest: <IP>",
        ),
        (
            "Received block blk_3587 of size 67108864 from /10.251.42.84",
            "Received block <BLK> of size <NUM> from <IP>",
        ),
        ("error code 0x1F after 3 retries", "error code <HEX> after <NUM> retries"),
        ("job_200811092030_0001 took 2.5 s", "job_200811092030_0001 took <NUM> s"),
    ],
)
def test_mask(message, expected):
    assert mask(message) == expected


def test_tokenize_splits_on_any_whitespace():
    assert tokenize("a  b\tc ") == ["a", "b", "c"]


def test_similar_messages_share_a_template_and_differences_become_wildcards():
    miner = TemplateMiner()
    a = miner.add("Deleting block blk_1 file /data/current/subdir1/blk_1")
    b = miner.add("Deleting block blk_2 file /data/current/subdir7/blk_2")
    assert a is b
    assert a.tokens == ["Deleting", "block", "<BLK>", "file", WILDCARD]
    assert a.count == 2


def test_different_lengths_or_first_words_never_merge():
    miner = TemplateMiner()
    first = miner.add("Verification succeeded for blk_5")
    assert miner.add("Verification succeeded for block blk_5") is not first
    assert miner.add("Validation succeeded for blk_5") is not first
    assert len(miner.templates) == 3


def test_threshold_controls_merging():
    strict = TemplateMiner(threshold=1.0)
    strict.add("user alice logged in")
    strict.add("user bob logged in")
    assert len(strict.templates) == 2
    loose = TemplateMiner(threshold=0.7)
    loose.add("user alice logged in")
    loose.add("user bob logged in")
    assert len(loose.templates) == 1


def test_match_does_not_learn():
    miner = TemplateMiner()
    template = miner.add("PacketResponder 1 for block blk_1 terminating")
    assert miner.match("PacketResponder 2 for block blk_9 terminating") is template
    assert miner.match("something entirely different") is None
    assert template.count == 1 and len(miner.templates) == 1


def test_invalid_threshold():
    with pytest.raises(ValueError):
        TemplateMiner(threshold=0.0)


def test_grouping_accuracy_on_hand_labelled_hdfs_sample(hdfs_2k_sample):
    with hdfs_2k_sample.open(newline="") as f:
        rows = list(csv.DictReader(f))
    miner = TemplateMiner()
    predicted = [miner.add(r["Content"]).id for r in rows]
    accuracy = grouping_accuracy(predicted, [r["EventId"] for r in rows])
    assert len(rows) == 2000
    assert accuracy >= 0.99  # measured: 0.9975
