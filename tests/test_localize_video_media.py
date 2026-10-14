from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from flowmind.skills import _media
from flowmind.skills import localize_video as lv


class LocalizeVideoMediaTests(TestCase):
    def test_subtitle_wrap_keeps_one_font_size_and_fits_lines(self) -> None:
        text = "A short line that needs to wrap into multiple readable subtitle lines"
        font_size, wrapped = lv._fit_subtitle(
            text, base_fs=32, play_res_w=500, margin_lr=40
        )

        self.assertEqual(font_size, 32)
        self.assertIn("\\N", wrapped)
        self.assertTrue(
            all(
                lv._est_text_width(line, font_size) <= 420
                for line in wrapped.split("\\N")
            )
        )

        with TemporaryDirectory() as tmp:
            ass_path = Path(
                lv._write_ass(
                    [
                        {"begin": 0, "end": 2, "text": text},
                        {"begin": 2, "end": 4, "text": "OK"},
                    ],
                    Path(tmp),
                    SimpleNamespace(subtitle_font_size=32),
                    width=1280,
                    height=720,
                )
            )
            ass = ass_path.read_text(encoding="utf-8")
            dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
            self.assertEqual(len(dialogue), 2)
            self.assertTrue(all("\\fs" not in line for line in dialogue))
            self.assertRegex(ass, re.compile(r"Style: Sub,[^,]+,32,"))

    def test_timed_audio_caps_tempo_and_normalizes_each_segment(self) -> None:
        commands: list[list[str]] = []

        with (
            TemporaryDirectory() as tmp,
            patch.object(_media, "probe_media", return_value=(4.0, 0, 0)),
            patch.object(
                _media,
                "run_ffmpeg",
                side_effect=lambda cmd: commands.append(cmd) or (0, "", ""),
            ),
        ):
            lv._build_timed_audio(
                [
                    {"begin": 0, "end": 1, "text": "first"},
                    {"begin": 1, "end": 2, "text": "second"},
                ],
                ["first.mp3", "second.mp3"],
                2,
                str(Path(tmp) / "dub.wav"),
            )

        conversions = [
            cmd
            for cmd in commands
            if cmd[-1].endswith(("dub_0000.wav", "dub_0001.wav"))
        ]
        self.assertEqual(len(conversions), 2)
        filters = [cmd[cmd.index("-filter:a") + 1] for cmd in conversions]
        self.assertTrue(all("loudnorm=I=-18:TP=-2:LRA=7" in f for f in filters))
        self.assertTrue(all("atempo=1.2500" in f for f in filters))

    def test_mix_keeps_original_background_and_limits_peaks(self) -> None:
        command: list[str] = []

        with (
            patch.object(_media.os.path, "exists", return_value=True),
            patch.object(
                _media,
                "run_ffmpeg",
                side_effect=lambda cmd: command.extend(cmd) or (0, "", ""),
            ),
        ):
            _media.mix_audio(
                "video.mp4",
                "dub.wav",
                "out.mp4",
                keep_background=True,
                bg_path="original.wav",
            )

        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("original.wav", command)
        self.assertIn("[2:a]volume=-12dB[bg]", graph)
        self.assertIn("alimiter=limit=0.95", graph)
        self.assertIn("[bg][dub]amix=", graph)
