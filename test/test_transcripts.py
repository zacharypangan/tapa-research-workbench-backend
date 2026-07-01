import unittest

from app.progress.transcripts import parse_transcript


class TranscriptParserTests(unittest.TestCase):
    def test_parses_vtt_timestamps_speakers_and_languages(self):
        cues = parse_transcript(
            """WEBVTT

00:00:01.000 --> 00:00:03.500
<v Kikusawa>時間情報には属性が必要です。</v>

cue-2
00:00:04.000 --> 00:00:05.000
Chair: We should preserve the original.
""",
            "meeting.vtt",
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].speaker, "Kikusawa")
        self.assertEqual(cues[0].start_seconds, 1.0)
        self.assertEqual(cues[0].language, "ja")
        self.assertEqual(cues[1].speaker, "Chair")
        self.assertEqual(cues[1].language, "en")

    def test_parses_srt_and_plain_text(self):
        srt = parse_transcript(
            """1
00:00:01,000 --> 00:00:02,000
GIS is an analytical interface.
""",
            "meeting.srt",
        )
        plain = parse_transcript("Speaker: First update.\n\n次の報告です。", "notes.txt")
        self.assertEqual(srt[0].end_seconds, 2.0)
        self.assertEqual(plain[0].speaker, "Speaker")
        self.assertEqual(plain[1].language, "ja")


if __name__ == "__main__":
    unittest.main()
