import unittest
from unittest.mock import patch

from flowmind.config import LocalizerConfig
from flowmind.skills import _inpaint
from flowmind.skills.localize_video import LocalizeVideoInput, _watermark_region


class WatermarkRemovalTests(unittest.TestCase):
    def test_local_asr_defaults_are_configured(self):
        cfg = LocalizerConfig()

        self.assertEqual(cfg.local_asr_model, "small")
        self.assertEqual(cfg.local_asr_device, "cuda")

    def test_bbox_is_validated_and_mapped_to_pixels(self):
        bbox = (0.8, 0.02, 0.95, 0.08)
        inp = LocalizeVideoInput(video_path="input.mp4", watermark_bbox=bbox)

        self.assertEqual(inp.watermark_bbox, bbox)
        self.assertEqual(
            _watermark_region(inp.watermark_bbox, 720, 1280),
            [{"x": 576, "y": 26, "w": 108, "h": 76}],
        )

    def test_invalid_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalizeVideoInput(
                video_path="input.mp4",
                watermark_bbox=(0.8, 0.2, 0.3, 0.4),
            )

    def test_solid_watermark_regions_reach_inpainting_adapter(self):
        solid_regions = [{"x": 576, "y": 26, "w": 108, "h": 76}]
        with patch.object(_inpaint, "_inpaint_frames_adapter", return_value="out.mp4") as run:
            result = _inpaint.erase_regions(
                "in.mp4", [], "out.mp4", "work", solid_regions=solid_regions,
            )

        self.assertEqual(result, "out.mp4")
        run.assert_called_once_with(
            "in.mp4", [], "out.mp4", "work", solid_regions=solid_regions,
        )


if __name__ == "__main__":
    unittest.main()
