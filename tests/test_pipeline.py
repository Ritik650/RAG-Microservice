from app.ingest.pipeline import point_id


def test_point_id_is_deterministic():
    assert point_id("doc.txt", 0) == point_id("doc.txt", 0)


def test_point_id_differs_by_index_and_source():
    assert point_id("doc.txt", 0) != point_id("doc.txt", 1)
    assert point_id("a.txt", 0) != point_id("b.txt", 0)


def test_point_id_is_uuid_shaped():
    pid = point_id("doc.txt", 3)
    assert pid.count("-") == 4 and len(pid) == 36
