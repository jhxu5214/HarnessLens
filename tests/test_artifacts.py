from harnesslens.core.artifacts import content_digest, read_json, write_json, write_text


def test_json_artifacts_are_atomic_and_canonical(tmp_path):
    path = write_json(tmp_path / "nested" / "value.json", {"b": 2, "a": [1]})

    assert read_json(path) == {"a": [1], "b": 2}
    assert content_digest({"b": 2, "a": [1]}) == content_digest({"a": [1], "b": 2})
    assert list(path.parent.glob("*.tmp")) == []


def test_text_artifacts_are_atomic(tmp_path):
    path = write_text(tmp_path / "nested" / "HEAD", "candidate\n")

    assert path.read_text() == "candidate\n"
    assert list(path.parent.glob("*.tmp")) == []
