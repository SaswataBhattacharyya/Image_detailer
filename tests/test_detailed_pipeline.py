import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from image_analyzer.cli.main import main as cli_main
from image_analyzer.config.settings import load_settings
from image_analyzer.detailed_pipeline import run_detailed_pipeline
from image_analyzer.providers.base import ProviderArtifact


def _visual_artifact(pass_key: str) -> ProviderArtifact:
    responses = {
        "composition_camera": "Low-angle beach image with horizon slightly below center and a wide landscape feel.",
        "objects_positions": "One small dark rock sits in the lower-right foreground.",
        "foreground_texture": "Wet reflective sand, fine ripples, shallow water channels, purple and pink reflections.",
        "middle_ground_horizon": "Calm ocean band with a soft surf line and clear horizon.",
        "sky_cloud_structure": "Wispy clouds radiate from the glowing horizon with darker clouds left and warmer clouds right.",
        "color_lighting": "Dominant blue, violet, pink, peach, and orange tones with vivid but natural saturation.",
        "negative_constraints": "no people, no buildings, no extra rocks, not panoramic, no visible circular sun disk",
    }
    return ProviderArtifact(provider="ollama", data={"content": responses[pass_key]})


class DetailedPipelineTests(unittest.TestCase):
    def test_run_detailed_pipeline_writes_run_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_settings(project_root)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (1536, 1024), (120, 90, 160)).save(image_path)

            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact(_pass_key_from_prompt(prompt)),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ):
                report = run_detailed_pipeline(image_path, config, output_root=tmp_path / "runs")

            run_dir = Path(report.run_dir)
            self.assertTrue((run_dir / "analysis" / "02_structured_scene_map.json").exists())
            self.assertTrue((run_dir / "analysis" / "03_refined_scene_map.json").exists())
            self.assertTrue((run_dir / "analysis" / "04_visual_hierarchy.json").exists())
            self.assertTrue((run_dir / "prompts" / "01_base_prompt.txt").exists())
            self.assertTrue((run_dir / "prompts" / "final_prompt.txt").exists())
            self.assertTrue((run_dir / "reports" / "final_prompt_package.json").exists())
            self.assertEqual(report.generation.status, "skipped")
            self.assertEqual(report.termination.reason, "skipped")
            self.assertEqual(len(report.iterations), 1)
            self.assertIn("single dark rock", report.prompt_package.final_prompt)

    def test_closed_loop_generation_stops_on_threshold(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_settings(project_root)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (512, 512), (120, 90, 160)).save(image_path)

            def fake_generation(**kwargs):
                output_image = kwargs["output_image"]
                Image.new("RGB", (512, 512), (120, 90, 160)).save(output_image)
                return kwargs["result_factory"](
                    enabled=True,
                    status="completed",
                    message="mock generation completed",
                    output_image=str(output_image),
                )

            comparison_payload = {
                "overall_similarity_score": 0.96,
                "semantic_similarity_score": 0.96,
                "issues": [],
                "negative_prompt_additions": [],
            }
            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact(_pass_key_from_prompt(prompt)),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.compare_images",
                return_value=ProviderArtifact(provider="ollama", data={"json": comparison_payload, "content": "{}"}),
            ), patch(
                "image_analyzer.detailed_pipeline._execute_generation_command",
                side_effect=lambda **kwargs: fake_generation(result_factory=_generation_result_factory, **kwargs),
            ):
                report = run_detailed_pipeline(
                    image_path,
                    config,
                    output_root=tmp_path / "runs",
                    enable_generation=True,
                    enable_comparison=True,
                    max_iterations=3,
                    target_score=0.90,
                )

            self.assertEqual(report.termination.reason, "threshold_met")
            self.assertEqual(report.termination.best_iteration, 1)
            self.assertGreaterEqual(report.termination.best_score, 0.90)
            self.assertEqual(len(report.iterations), 1)
            self.assertEqual(report.generation.status, "completed")

    def test_cli_analyze_detailed_prints_run_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (800, 600), (100, 110, 180)).save(image_path)

            stdout = io.StringIO()
            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact(_pass_key_from_prompt(prompt)),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch("sys.argv", ["image-analyzer", "analyze-detailed", str(image_path), "--output-dir", str(tmp_path / "runs")]), patch(
                "sys.stdout",
                stdout,
            ):
                cli_main()

            self.assertIn("Detailed run directory:", stdout.getvalue())

    def test_generation_requested_without_backend_reports_config_failure(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_settings(project_root)
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "beach.png"
            Image.new("RGB", (512, 512), (120, 90, 160)).save(image_path)

            with patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.analyze_visual_pass",
                side_effect=lambda image_path, prompt, num_predict=900: _visual_artifact(_pass_key_from_prompt(prompt)),
            ), patch(
                "image_analyzer.providers.ollama.OllamaSynthesisProvider.generate_json",
                return_value=ProviderArtifact(provider="ollama", data={"json": {}}),
            ), patch.dict("os.environ", {"IMAGE_ANALYZER_QWEN_IMAGE_RUNNER": ""}, clear=False):
                report = run_detailed_pipeline(
                    image_path,
                    config,
                    output_root=tmp_path / "runs",
                    enable_generation=True,
                    enable_comparison=False,
                    max_iterations=1,
                )

            self.assertEqual(report.generation.status, "failed_runtime")


def _pass_key_from_prompt(prompt: str) -> str:
    if "composition, framing" in prompt:
        return "composition_camera"
    if "List main objects" in prompt:
        return "objects_positions"
    if "lower part of the image" in prompt:
        return "foreground_texture"
    if "middle-ground structure" in prompt:
        return "middle_ground_horizon"
    if "upper image cloud structure" in prompt:
        return "sky_cloud_structure"
    if "Describe palette" in prompt:
        return "color_lighting"
    return "negative_constraints"


def _generation_result_factory(**kwargs):
    from image_analyzer.models.schemas import GenerationResult

    return GenerationResult(**kwargs)


if __name__ == "__main__":
    unittest.main()
