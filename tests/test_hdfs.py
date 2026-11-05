import numpy as np

from logfusion.hdfs import BlockEvents, block_ids, build_block_events, load_labels, parse_line

LINES = [
    "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1 src: /10.0.0.1:1 dest: /10.0.0.2:2\n",
    "081109 203519 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /tmp/a. blk_1\n",
    "081109 203520 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-2 src: /10.0.0.3:1 dest: /10.0.0.4:2\n",
    "081109 203521 12 INFO dfs.FSNamesystem: BLOCK* ask 10.0.0.1:50010 to replicate blk_1 to datanode(s) 10.0.0.9:50010\n",
    "this line is malformed\n",
    "081109 203522 9 WARN dfs.DataNode: Deleting block blk_-2 file /x/blk_-2\n",
]


def test_parse_line_fields():
    record = parse_line(LINES[0])
    assert record is not None
    assert (record.date, record.time, record.pid, record.level) == ("081109", "203518", 143, "INFO")
    assert record.component == "dfs.DataNode$DataXceiver"
    assert record.content.startswith("Receiving block blk_1")
    assert parse_line("garbage") is None


def test_block_ids_are_distinct_and_ordered():
    assert block_ids("copy blk_5 to blk_-7 then blk_5 again") == ["blk_5", "blk_-7"]


def test_counts_per_block():
    events = build_block_events(LINES)
    assert events.lines == 6 and events.malformed == 1
    assert events.blocks == ["blk_1", "blk_-2"]
    # templates: 0 Receiving, 1 allocateBlock, 2 ask..replicate, 3 Deleting
    assert len(events.templates) == 4
    np.testing.assert_array_equal(events.counts, [[1, 1, 1, 0], [1, 0, 0, 1]])


def test_save_and_load_round_trip(tmp_path):
    events = build_block_events(LINES)
    path = tmp_path / "events.npz"
    events.save(path)
    loaded = BlockEvents.load(path)
    assert loaded.blocks == events.blocks and loaded.templates == events.templates
    assert (loaded.lines, loaded.malformed) == (6, 1)
    np.testing.assert_array_equal(loaded.counts, events.counts)


def test_load_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("BlockId,Label\nblk_1,Normal\nblk_-2,Anomaly\n")
    assert load_labels(path) == {"blk_1": 0, "blk_-2": 1}
