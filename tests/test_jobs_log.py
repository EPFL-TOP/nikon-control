from nikon_control.jobs_log import log, log_path


def test_log_writes_to_configured_path(tmp_path, monkeypatch, capsys):
    p = tmp_path / "x.log"
    monkeypatch.setenv("NIKON_CONTROL_LOG", str(p))

    assert log_path() == p

    log("hello world")

    contents = p.read_text(encoding="utf-8")
    assert "hello world" in contents
    assert "T" in contents.split()[0]  # ISO timestamp prefix

    captured = capsys.readouterr()
    assert "hello world" in captured.out


def test_log_creates_parent_dir(tmp_path, monkeypatch):
    p = tmp_path / "nested" / "dirs" / "x.log"
    monkeypatch.setenv("NIKON_CONTROL_LOG", str(p))

    log("first line")

    assert p.exists()
    assert "first line" in p.read_text(encoding="utf-8")
