"""Tests for graceful yt-dlp error handling."""

import sys
from unittest import mock

import pytest


# Create proper exception hierarchy that subtitle_video will import
class _DownloadError(Exception):
    pass


class _ExtractorError(Exception):
    pass


# Build mock yt_dlp with real exception classes
_mock_yt_dlp = mock.MagicMock()
_mock_utils = mock.MagicMock()
_mock_utils.DownloadError = _DownloadError
_mock_utils.ExtractorError = _ExtractorError
_mock_yt_dlp.utils = _mock_utils

sys.modules["yt_dlp"] = _mock_yt_dlp
sys.modules["yt_dlp.utils"] = _mock_utils

# Mock other heavy deps
for mod in [
    "whisper",
    "moviepy",
    "moviepy.video",
    "moviepy.video.tools",
    "moviepy.video.tools.subtitles",
    "torch",
    "pydub",
]:
    sys.modules.setdefault(mod, mock.MagicMock())

# Force reimport
if "subtitle_video" in sys.modules:
    del sys.modules["subtitle_video"]

from subtitle_video import download_youtube_video, VideoDownloadError


def _setup_mock_download(side_effect):
    """Configure mock YoutubeDL to raise given exception on download."""
    mock_ydl_instance = mock.MagicMock()
    mock_ydl_instance.download.side_effect = side_effect
    mock_ydl_instance.__enter__ = mock.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = mock.Mock(return_value=False)
    _mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance


class TestDownloadYoutubeVideo:
    def test_private_video_error(self):
        _setup_mock_download(_DownloadError("Video unavailable: private video"))
        with pytest.raises(VideoDownloadError, match="unavailable or private"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_geo_blocked_error(self):
        _setup_mock_download(_DownloadError("not available in your country due to geo restriction"))
        with pytest.raises(VideoDownloadError, match="geo-blocked"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_age_restricted_error(self):
        _setup_mock_download(_DownloadError("Sign in to confirm your age"))
        with pytest.raises(VideoDownloadError, match="age-restricted"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_network_error(self):
        _setup_mock_download(_DownloadError("urlopen error [Errno -3] Temporary failure"))
        with pytest.raises(VideoDownloadError, match="Network error"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_generic_download_error(self):
        _setup_mock_download(_DownloadError("Something else went wrong"))
        with pytest.raises(VideoDownloadError, match="Failed to download"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_extractor_error(self):
        _setup_mock_download(_ExtractorError("Could not extract"))
        with pytest.raises(VideoDownloadError, match="Could not extract video info"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_unexpected_error(self):
        _setup_mock_download(RuntimeError("something unexpected"))
        with pytest.raises(VideoDownloadError, match="Unexpected error"):
            download_youtube_video("https://youtube.com/watch?v=test", {}, {})

    def test_success(self):
        mock_ydl_instance = mock.MagicMock()
        mock_ydl_instance.download.return_value = None
        mock_ydl_instance.__enter__ = mock.Mock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = mock.Mock(return_value=False)
        _mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
        # Should not raise
        download_youtube_video("https://youtube.com/watch?v=test", {}, {})
