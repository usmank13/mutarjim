"""Tests for audio chunking and transcription merging."""
import os
import shutil
import sys
from unittest import mock

import pytest

# Mock heavy deps
for mod in ['whisper', 'moviepy', 'moviepy.video', 'moviepy.video.tools',
            'moviepy.video.tools.subtitles', 'yt_dlp', 'yt_dlp.utils', 'torch']:
    sys.modules.setdefault(mod, mock.MagicMock())

# We need pydub to be real for AudioSegment tests, but if not installed, mock it
try:
    import pydub
    HAS_PYDUB = True
except ImportError:
    sys.modules.setdefault('pydub', mock.MagicMock())
    HAS_PYDUB = False

from subtitle_video import merge_chunked_results, split_audio_into_chunks


class TestMergeChunkedResults:
    def test_single_chunk_passthrough(self):
        result = {'text': 'hello', 'segments': [
            {'start': 0, 'end': 5, 'text': 'hello'}
        ]}
        merged = merge_chunked_results([(result, 0.0)])
        assert merged is result  # Should return the same object

    def test_two_chunks_no_overlap(self):
        r1 = {'text': 'hello', 'segments': [
            {'start': 0, 'end': 5, 'text': 'hello'},
            {'start': 5, 'end': 10, 'text': 'world'},
        ]}
        r2 = {'text': 'foo', 'segments': [
            {'start': 0, 'end': 5, 'text': 'foo'},
            {'start': 5, 'end': 10, 'text': 'bar'},
        ]}
        merged = merge_chunked_results([(r1, 0.0), (r2, 100.0)])
        assert len(merged['segments']) == 4
        assert merged['segments'][0]['start'] == 0
        assert merged['segments'][2]['start'] == 100.0
        assert merged['segments'][3]['end'] == 110.0

    def test_overlapping_segments_deduplicated(self):
        r1 = {'text': 'a b', 'segments': [
            {'start': 0, 'end': 5, 'text': 'a'},
            {'start': 5, 'end': 10, 'text': 'b'},
            {'start': 25, 'end': 30, 'text': 'overlap'},
        ]}
        # Second chunk starts at offset 28s, so its segments overlap with the last segment of r1
        r2 = {'text': 'overlap c', 'segments': [
            {'start': 0, 'end': 5, 'text': 'overlap'},  # offset=28 -> 28-33, overlaps with 25-30
            {'start': 5, 'end': 10, 'text': 'c'},
        ]}
        merged = merge_chunked_results([(r1, 0.0), (r2, 28.0)])
        # The overlapping segment (28-33) should be deduplicated with (25-30)
        texts = [s['text'] for s in merged['segments']]
        assert texts == ['a', 'b', 'overlap', 'c']

    def test_empty_segments(self):
        r1 = {'text': '', 'segments': []}
        r2 = {'text': 'hello', 'segments': [
            {'start': 0, 'end': 5, 'text': 'hello'}
        ]}
        merged = merge_chunked_results([(r1, 0.0), (r2, 100.0)])
        assert len(merged['segments']) == 1

    def test_merged_text(self):
        r1 = {'text': 'hello', 'segments': [
            {'start': 0, 'end': 5, 'text': ' hello '}
        ]}
        r2 = {'text': 'world', 'segments': [
            {'start': 0, 'end': 5, 'text': ' world '}
        ]}
        merged = merge_chunked_results([(r1, 0.0), (r2, 100.0)])
        assert merged['text'] == 'hello world'


@pytest.mark.skipif(not HAS_PYDUB or shutil.which('ffmpeg') is None, reason="pydub not installed or ffmpeg not available")
class TestSplitAudioIntoChunks:
    def test_short_audio_no_split(self, tmp_path):
        """Audio shorter than chunk duration should return single chunk."""
        from pydub.generators import Sine
        # Generate 10 seconds of audio
        audio = Sine(440).to_audio_segment(duration=10000)
        audio_path = str(tmp_path / "short.mp3")
        audio.export(audio_path, format="mp3")
        
        chunks = split_audio_into_chunks(audio_path, chunk_duration_ms=30*60*1000)
        assert len(chunks) == 1
        assert chunks[0] == (audio_path, 0.0)

    def test_long_audio_splits(self, tmp_path):
        """Audio longer than chunk duration should be split."""
        from pydub.generators import Sine
        # Generate 5 minutes of audio, split at 2 minute chunks
        audio = Sine(440).to_audio_segment(duration=5*60*1000)
        audio_path = str(tmp_path / "long.mp3")
        audio.export(audio_path, format="mp3")
        
        chunks = split_audio_into_chunks(audio_path, chunk_duration_ms=2*60*1000, overlap_ms=10*1000)
        assert len(chunks) >= 3
        # First chunk starts at 0
        assert chunks[0][1] == 0.0
        # Chunk files should exist
        for path, offset in chunks:
            assert os.path.exists(path)
        # Offsets should be increasing
        offsets = [c[1] for c in chunks]
        assert offsets == sorted(offsets)
