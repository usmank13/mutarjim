"""Tests for subtitle_video.py — unit tests that don't require video files or API keys."""

import sys
from unittest import mock

import pytest
import pandas as pd

# Mock heavy dependencies before importing subtitle_video
sys.modules.setdefault("whisper", mock.MagicMock())
sys.modules.setdefault("moviepy", mock.MagicMock())
sys.modules.setdefault("moviepy.video", mock.MagicMock())
sys.modules.setdefault("moviepy.video.tools", mock.MagicMock())
sys.modules.setdefault("moviepy.video.tools.subtitles", mock.MagicMock())
sys.modules.setdefault("yt_dlp", mock.MagicMock())

from subtitle_video import (
    _format_timedelta,
    create_subtitles_df,
    export_subtitles,
    load_config,
)


class TestFormatTimedelta:
    def test_zero(self):
        assert _format_timedelta(0, ",") == "00:00:00,000"

    def test_simple_seconds(self):
        assert _format_timedelta(5, ",") == "00:00:05,000"

    def test_minutes_and_seconds(self):
        assert _format_timedelta(125, ",") == "00:02:05,000"

    def test_hours(self):
        assert _format_timedelta(3661, ",") == "01:01:01,000"

    def test_vtt_separator(self):
        assert _format_timedelta(10, ".") == "00:00:10.000"

    def test_fractional_seconds(self):
        result = _format_timedelta(1.5, ",")
        assert result == "00:00:01,500"


class TestCreateSubtitlesDf:
    def test_basic(self):
        result = {
            "text": "Hello world",
            "segments": [
                {"start": 0, "end": 5, "text": "Hello"},
                {"start": 5, "end": 10, "text": "world"},
            ],
        }
        df = create_subtitles_df(result)
        assert len(df) == 2
        assert list(df.columns) == ["start", "end", "text"]
        assert df.iloc[0]["text"] == "Hello"
        assert df.iloc[1]["start"] == 5

    def test_empty_segments(self):
        result = {"text": "", "segments": []}
        df = create_subtitles_df(result)
        assert len(df) == 0


class TestExportSubtitles:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "start": [0, 5, 12],
                "end": [5, 12, 20],
                "text": ["First line", "Second line", "Third line"],
            }
        )

    def test_srt_export(self, sample_df, tmp_path):
        output = tmp_path / "test.srt"
        export_subtitles(sample_df, "srt", str(output))
        content = output.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:05,000" in content
        assert "First line" in content
        assert "00:00:12,000 --> 00:00:20,000" in content

    def test_vtt_export(self, sample_df, tmp_path):
        output = tmp_path / "test.vtt"
        export_subtitles(sample_df, "vtt", str(output))
        content = output.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:05.000" in content

    def test_invalid_format(self, sample_df, tmp_path):
        with pytest.raises(ValueError, match="Unsupported subtitle format"):
            export_subtitles(sample_df, "ass", str(tmp_path / "test.ass"))

    def test_srt_sequential_numbering(self, sample_df, tmp_path):
        output = tmp_path / "test.srt"
        export_subtitles(sample_df, "srt", str(output))
        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # SRT format: number, timestamp, text, blank
        assert lines[0] == "1"
        assert lines[4] == "2"
        assert lines[8] == "3"


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("name: test\ndownload: false\nmodel_type: base\n")
        config = load_config(str(cfg))
        assert config["name"] == "test"
        assert config["download"] is False

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
