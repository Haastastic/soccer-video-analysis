"""publish_site.py: opponent name and logo from a game's video file. Made-up school names only."""

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "webapp"), str(Path(__file__).resolve().parents[1])]

from db import MemoryDB  # noqa: E402

import publish_site as p  # noqa: E402


def test_opponent_from_video():
    assert p.opponent_from_video("Hometown JV vs Northfield JV 2026-01-02.mp4") == "Northfield JV"
    assert p.opponent_from_video("Hometown JV vs. Lake Ridge-C 2026-01-02.mp4") == "Lake Ridge C"
    assert p.opponent_from_video("Hometown Varsity vs St. Elm Academy Varsity 2026-01-02.mp4") == "St. Elm Academy"
    assert p.opponent_from_video("Hometown vs Northfield.mp4") is None  # no date
    assert p.opponent_from_video("Envoy Cup 2026-01-02.mp4") is None  # "vs" must be a word


def logo(path: Path) -> Path:
    buf = BytesIO()
    Image.new("RGB", (40, 40), (10, 20, 30)).save(buf, "JPEG")
    path.write_bytes(buf.getvalue())
    return path


def test_find_logo(tmp_path):
    logo(tmp_path / "St Elm Logo.jfif")
    logo(tmp_path / "St Elm Academy Logo.png")
    logo(tmp_path / "Lake Logo.jpg")
    (tmp_path / "Northfield Logo.txt").write_text("not a logo")
    assert p.find_logo(tmp_path, "St. Elm Academy").name == "St Elm Academy Logo.png"  # the longest school wins
    assert p.find_logo(tmp_path, "St. Elm JV").name == "St Elm Logo.jfif"
    assert p.find_logo(tmp_path, "Lake Ridge C").name == "Lake Logo.jpg"
    assert p.find_logo(tmp_path, "Northfield JV") is None
    assert p.find_logo(tmp_path, "Lakeside") is None  # whole words only


def test_set_opponents_keeps_admin_values(tmp_path, monkeypatch):
    video = tmp_path / "Hometown JV vs Northfield-JV 2026-01-02.mp4"
    other = tmp_path / "Hometown JV vs Lake Ridge JV 2026-01-09.mp4"
    logo(tmp_path / "Northfield Logo.jpg")
    logo(tmp_path / "Lake Ridge Logo.jpg")
    monkeypatch.setattr(p, "game_videos", lambda: {("jv", "2026-01-02"): video, ("jv", "2026-01-09"): other})
    db = MemoryDB()
    db.put_game("jv_2026-01-09", {"opponent": "Set by an admin"})
    p.set_opponents(db, {"jv": ["2026-01-02", "2026-01-09", "2026-01-16"]})
    new = db.get_game("jv_2026-01-02")
    assert new["opponent"] == "Northfield JV"
    assert Image.open(BytesIO(new["logo"])).format == "PNG"  # re-encoded like an admin upload
    kept = db.get_game("jv_2026-01-09")
    assert kept["opponent"] == "Set by an admin" and "logo" in kept  # only missing fields are filled
    assert db.get_game("jv_2026-01-16") == {}  # no video known: untouched


def test_clip_files_only_named_clips(tmp_path, monkeypatch):
    """Clips go only by name from the export's clips.json, and a name off the pattern stops the upload."""
    import json

    import pytest

    import site_clips

    export, clips = tmp_path / "export", tmp_path / "clips"
    monkeypatch.setattr(p, "EXPORT", export)
    monkeypatch.setattr(site_clips, "CLIPS", clips)
    d = export / "jv" / "games" / "2026-01-02"
    d.mkdir(parents=True)
    (clips / "jv" / "2026-01-02").mkdir(parents=True)
    for name in ["41_100000.mp4", "42_5.mp4"]:
        (clips / "jv" / "2026-01-02" / name).write_bytes(b"x")
    (d / "clips.json").write_text(json.dumps({"41": [["Longest", [[1.0, "a", "41_100000.mp4"]]]]}))
    assert [path for path, _ in p.clip_files()] == ["clips/jv/2026-01-02/41_100000.mp4"]  # 42's is not named
    (d / "clips.json").write_text(json.dumps({"41": [["Longest", [[1.0, "a", "../../secret.mp4"]]]]}))
    with pytest.raises(SystemExit):
        p.clip_files()
